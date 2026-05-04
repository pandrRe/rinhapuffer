#define _GNU_SOURCE
#include "ring.h"

#include <errno.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef IORING_FEAT_SINGLE_MMAP
#define IORING_FEAT_SINGLE_MMAP (1U << 0)
#endif

static inline int sys_io_uring_setup(unsigned entries, struct io_uring_params *p) {
	return (int)syscall(SYS_io_uring_setup, entries, p);
}

static inline int sys_io_uring_enter(int fd, unsigned to_submit,
                                     unsigned min_complete, unsigned flags) {
	return (int)syscall(SYS_io_uring_enter, fd, to_submit, min_complete, flags,
	                    NULL, 0);
}

int ring_init(struct ring *r, unsigned entries) {
	memset(r, 0, sizeof *r);

	struct io_uring_params p;
	memset(&p, 0, sizeof p);

	// SINGLE_ISSUER + DEFER_TASKRUN + COOP_TASKRUN: matches the rinhapuffer
	// API server's setup. SINGLE_ISSUER tells the kernel exactly one task
	// submits; DEFER_TASKRUN coalesces task-work onto io_uring_enter (cuts
	// wakeup churn); COOP_TASKRUN avoids IPIs on cross-core completions.
	p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN |
	          IORING_SETUP_COOP_TASKRUN;

	int fd = sys_io_uring_setup(entries, &p);
	if (fd < 0) {
		// Older kernel: retry with no flags.
		memset(&p, 0, sizeof p);
		fd = sys_io_uring_setup(entries, &p);
		if (fd < 0) return -1;
	}
	r->fd = fd;
	r->features = p.features;

	size_t sring_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
	size_t cring_size = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
	int single_mmap = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
	if (single_mmap) {
		if (cring_size > sring_size) sring_size = cring_size;
		cring_size = sring_size;
	}

	void *sq_ptr = mmap(NULL, sring_size, PROT_READ | PROT_WRITE,
	                    MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
	if (sq_ptr == MAP_FAILED) goto fail;
	r->sq_mmap = sq_ptr;
	r->sq_mmap_size = sring_size;

	void *cq_ptr;
	if (single_mmap) {
		cq_ptr = sq_ptr;
		r->cq_mmap = NULL;
	} else {
		cq_ptr = mmap(NULL, cring_size, PROT_READ | PROT_WRITE,
		              MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
		if (cq_ptr == MAP_FAILED) goto fail;
		r->cq_mmap = cq_ptr;
		r->cq_mmap_size = cring_size;
	}

	r->sq_khead = (unsigned *)((char *)sq_ptr + p.sq_off.head);
	r->sq_ktail = (unsigned *)((char *)sq_ptr + p.sq_off.tail);
	r->sq_ring_mask = *(unsigned *)((char *)sq_ptr + p.sq_off.ring_mask);
	r->sq_entries = *(unsigned *)((char *)sq_ptr + p.sq_off.ring_entries);
	r->sq_array = (unsigned *)((char *)sq_ptr + p.sq_off.array);

	r->cq_khead = (unsigned *)((char *)cq_ptr + p.cq_off.head);
	r->cq_ktail = (unsigned *)((char *)cq_ptr + p.cq_off.tail);
	r->cq_ring_mask = *(unsigned *)((char *)cq_ptr + p.cq_off.ring_mask);
	r->cqes = (struct io_uring_cqe *)((char *)cq_ptr + p.cq_off.cqes);

	size_t sqes_size = p.sq_entries * sizeof(struct io_uring_sqe);
	void *sqes_ptr = mmap(NULL, sqes_size, PROT_READ | PROT_WRITE,
	                      MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
	if (sqes_ptr == MAP_FAILED) goto fail;
	r->sqes = (struct io_uring_sqe *)sqes_ptr;
	r->sqes_mmap_size = sqes_size;

	r->sq_local_tail = atomic_load_explicit(
	    (_Atomic unsigned *)r->sq_ktail, memory_order_relaxed);

	return 0;

fail:
	ring_deinit(r);
	return -1;
}

void ring_deinit(struct ring *r) {
	if (r->sqes) munmap(r->sqes, r->sqes_mmap_size);
	if (r->cq_mmap) munmap(r->cq_mmap, r->cq_mmap_size);
	if (r->sq_mmap) munmap(r->sq_mmap, r->sq_mmap_size);
	if (r->fd > 0) close(r->fd);
	memset(r, 0, sizeof *r);
}

struct io_uring_sqe *ring_get_sqe(struct ring *r) {
	unsigned head = atomic_load_explicit((_Atomic unsigned *)r->sq_khead,
	                                     memory_order_acquire);
	if (r->sq_local_tail - head >= r->sq_entries) return NULL;
	unsigned idx = r->sq_local_tail & r->sq_ring_mask;
	struct io_uring_sqe *sqe = &r->sqes[idx];
	memset(sqe, 0, sizeof *sqe);
	r->sq_array[idx] = idx;
	r->sq_local_tail++;
	return sqe;
}

int ring_submit_and_wait(struct ring *r, unsigned wait_nr) {
	unsigned old_ktail = atomic_load_explicit(
	    (_Atomic unsigned *)r->sq_ktail, memory_order_relaxed);
	unsigned to_submit = r->sq_local_tail - old_ktail;
	if (to_submit > 0) {
		atomic_store_explicit((_Atomic unsigned *)r->sq_ktail,
		                      r->sq_local_tail, memory_order_release);
	}

	if (to_submit == 0 && wait_nr == 0) return 0;

	unsigned flags = wait_nr ? IORING_ENTER_GETEVENTS : 0;
	int rc;
	do {
		rc = sys_io_uring_enter(r->fd, to_submit, wait_nr, flags);
	} while (rc < 0 && errno == EINTR);
	return rc;
}

unsigned ring_cq_ready(const struct ring *r) {
	unsigned tail = atomic_load_explicit((const _Atomic unsigned *)r->cq_ktail,
	                                     memory_order_acquire);
	unsigned head = *r->cq_khead;
	return tail - head;
}

struct io_uring_cqe *ring_cq_peek(const struct ring *r, unsigned i) {
	unsigned head = *r->cq_khead;
	return &r->cqes[(head + i) & r->cq_ring_mask];
}

void ring_cq_advance(struct ring *r, unsigned n) {
	atomic_store_explicit((_Atomic unsigned *)r->cq_khead,
	                      *r->cq_khead + n, memory_order_release);
}

void prep_accept_multishot(struct io_uring_sqe *sqe, int fd, uint64_t ud) {
	memset(sqe, 0, sizeof *sqe);
	sqe->opcode = IORING_OP_ACCEPT;
	sqe->fd = fd;
	sqe->user_data = ud;
	sqe->ioprio = IORING_ACCEPT_MULTISHOT;
}

void prep_connect(struct io_uring_sqe *sqe, int fd,
                  const struct sockaddr *addr, socklen_t len, uint64_t ud) {
	memset(sqe, 0, sizeof *sqe);
	sqe->opcode = IORING_OP_CONNECT;
	sqe->fd = fd;
	sqe->addr = (uintptr_t)addr;
	sqe->off = (uint64_t)len;
	sqe->user_data = ud;
}

void prep_poll_add(struct io_uring_sqe *sqe, int fd, unsigned poll_mask, uint64_t ud) {
	memset(sqe, 0, sizeof *sqe);
	sqe->opcode = IORING_OP_POLL_ADD;
	sqe->fd = fd;
	sqe->poll32_events = poll_mask;
	sqe->user_data = ud;
}

void prep_close(struct io_uring_sqe *sqe, int fd, uint64_t ud) {
	memset(sqe, 0, sizeof *sqe);
	sqe->opcode = IORING_OP_CLOSE;
	sqe->fd = fd;
	sqe->user_data = ud;
}

void prep_sendmsg(struct io_uring_sqe *sqe, int fd, struct msghdr *msg,
                  unsigned msg_flags, uint64_t ud) {
	memset(sqe, 0, sizeof *sqe);
	sqe->opcode = IORING_OP_SENDMSG;
	sqe->fd = fd;
	sqe->addr = (uintptr_t)msg;
	sqe->len = 1;          // msg_iovlen — only one iovec in our handoff frame
	sqe->msg_flags = msg_flags;
	sqe->user_data = ud;
}

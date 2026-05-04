#ifndef RINHALB_RING_H
#define RINHALB_RING_H

#define _GNU_SOURCE
#include <linux/io_uring.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/socket.h>

struct ring {
	int fd;

	unsigned *sq_khead;
	unsigned *sq_ktail;
	unsigned  sq_ring_mask;
	unsigned  sq_entries;
	unsigned *sq_array;
	struct io_uring_sqe *sqes;
	unsigned  sq_local_tail;

	unsigned *cq_khead;
	unsigned *cq_ktail;
	unsigned  cq_ring_mask;
	struct io_uring_cqe *cqes;

	unsigned  features;

	void *sq_mmap;
	size_t sq_mmap_size;
	void *cq_mmap;
	size_t cq_mmap_size;
	size_t sqes_mmap_size;
};

int ring_init(struct ring *r, unsigned entries);
void ring_deinit(struct ring *r);

struct io_uring_sqe *ring_get_sqe(struct ring *r);
int ring_submit_and_wait(struct ring *r, unsigned wait_nr);

unsigned ring_cq_ready(const struct ring *r);
struct io_uring_cqe *ring_cq_peek(const struct ring *r, unsigned i);
void ring_cq_advance(struct ring *r, unsigned n);

void prep_accept_multishot(struct io_uring_sqe *sqe, int fd, uint64_t ud);
void prep_connect(struct io_uring_sqe *sqe, int fd,
                  const struct sockaddr *addr, socklen_t len, uint64_t ud);
void prep_poll_add(struct io_uring_sqe *sqe, int fd, unsigned poll_mask, uint64_t ud);
void prep_close(struct io_uring_sqe *sqe, int fd, uint64_t ud);

#endif

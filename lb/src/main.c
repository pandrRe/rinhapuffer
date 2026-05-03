// rinhalb — round-robin TCP→UDS L4 splicer.
//
// Single thread. io_uring drives readiness; splice(2) does the byte movement
// through per-conn pipes (zero-copy). No allocation past startup.

#define _GNU_SOURCE
#include "ring.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// ── config ─────────────────────────────────────────────────────────────────

#define MAX_CONNS     4096
#define MAX_BACKENDS  8
#define LISTEN_BACKLOG 4096
#define SQ_DEPTH      4096
#define CQE_BATCH     256
#define SPLICE_CHUNK  (16 * 1024)
#define PIPE_SZ       (64 * 1024)

// ── user_data layout ───────────────────────────────────────────────────────
//   bits 0..15  slot
//   bits 16..23 op tag
//   bits 24..31 generation
// Stale CQEs (slot reused before CQE drained) are dropped via gen mismatch.

enum {
	OP_ACCEPT  = 0,
	OP_CONNECT = 1,
	OP_POLL_C  = 2,  // poll on client fd
	OP_POLL_B  = 3,  // poll on backend fd
};

static inline uint64_t mk_ud(uint16_t slot, uint8_t op, uint8_t gen) {
	return ((uint64_t)gen << 24) | ((uint64_t)op << 16) | (uint64_t)slot;
}
static inline uint16_t ud_slot(uint64_t ud) { return (uint16_t)(ud & 0xffff); }
static inline uint8_t  ud_op(uint64_t ud)   { return (uint8_t)((ud >> 16) & 0xff); }
static inline uint8_t  ud_gen(uint64_t ud)  { return (uint8_t)((ud >> 24) & 0xff); }

// ── per-direction state ────────────────────────────────────────────────────

enum dir_state {
	DIR_DEAD = 0,    // not initialised / fully closed
	DIR_OPEN,        // splicing live
	DIR_SRC_EOF,     // src EOFed; pipe may still have bytes to drain
};

// ── connection slab ────────────────────────────────────────────────────────

typedef struct {
	int       fd_client;
	int       fd_backend;
	int       pipe_c2b[2];     // [0]=rd, [1]=wr
	int       pipe_b2c[2];

	uint8_t   in_use;
	uint8_t   gen;             // bumped each accept-init; tags every SQE

	uint8_t   c2b_state;
	uint8_t   b2c_state;

	int       c2b_in_pipe;     // bytes currently buffered in c2b pipe
	int       b2c_in_pipe;

	uint8_t   poll_c_armed;
	uint8_t   poll_b_armed;
	unsigned  poll_c_mask;     // events POLLIN/POLLOUT we last armed
	unsigned  poll_b_mask;
} conn_t;

static conn_t conns[MAX_CONNS];

static int free_stack[MAX_CONNS];
static int free_top;

static int alloc_slot(void) {
	if (free_top == 0) return -1;
	return free_stack[--free_top];
}

static void return_slot(int slot) {
	free_stack[free_top++] = slot;
}

// ── backends ───────────────────────────────────────────────────────────────

static struct sockaddr_un backend_addrs[MAX_BACKENDS];
static int                n_backends = 0;
static uint64_t           rr_counter = 0;

// ── io_uring ───────────────────────────────────────────────────────────────

static struct ring R;

// ── helpers ────────────────────────────────────────────────────────────────

static int set_nonblock(int fd) {
	int fl = fcntl(fd, F_GETFL, 0);
	if (fl < 0) return -1;
	return fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

static void close_fd(int *fd) {
	if (*fd >= 0) { close(*fd); *fd = -1; }
}

static void teardown_conn(int slot) {
	conn_t *c = &conns[slot];
	close_fd(&c->fd_client);
	close_fd(&c->fd_backend);
	close_fd(&c->pipe_c2b[0]);
	close_fd(&c->pipe_c2b[1]);
	close_fd(&c->pipe_b2c[0]);
	close_fd(&c->pipe_b2c[1]);
	c->in_use = 0;
	c->c2b_state = DIR_DEAD;
	c->b2c_state = DIR_DEAD;
	c->c2b_in_pipe = 0;
	c->b2c_in_pipe = 0;
	c->poll_c_armed = 0;
	c->poll_b_armed = 0;
	c->gen++;  // invalidate any in-flight CQEs targeting this slot
	return_slot(slot);
}

// Forward declarations for the splice driver.
static void arm_poll_client(int slot, unsigned mask);
static void arm_poll_backend(int slot, unsigned mask);

// drain one direction. Returns 0 if direction still alive, -1 if conn died.
static int drive_dir(int slot, int dir_c2b) {
	conn_t *c = &conns[slot];
	int      src       = dir_c2b ? c->fd_client    : c->fd_backend;
	int      dst       = dir_c2b ? c->fd_backend   : c->fd_client;
	int     *pipe_p    = dir_c2b ? c->pipe_c2b     : c->pipe_b2c;
	int     *in_pipe   = dir_c2b ? &c->c2b_in_pipe : &c->b2c_in_pipe;
	uint8_t *state     = dir_c2b ? &c->c2b_state   : &c->b2c_state;

	if (*state == DIR_DEAD) return 0;

	for (;;) {
		// 1) Pull from src into pipe (only while src still alive).
		if (*state == DIR_OPEN) {
			ssize_t n = splice(src, NULL, pipe_p[1], NULL,
			                   SPLICE_CHUNK,
			                   SPLICE_F_MOVE | SPLICE_F_NONBLOCK);
			if (n > 0) {
				*in_pipe += (int)n;
			} else if (n == 0) {
				// src EOF
				*state = DIR_SRC_EOF;
			} else {
				if (errno == EAGAIN || errno == EWOULDBLOCK) {
					// nothing more to read right now
				} else if (errno == EINTR) {
					continue;
				} else {
					return -1;
				}
			}
		}

		// 2) Push pipe into dst.
		if (*in_pipe > 0) {
			ssize_t n = splice(pipe_p[0], NULL, dst, NULL,
			                   *in_pipe,
			                   SPLICE_F_MOVE | SPLICE_F_NONBLOCK);
			if (n > 0) {
				*in_pipe -= (int)n;
				if (*in_pipe > 0) {
					// dst can't take any more; wait for POLLOUT.
					if (dir_c2b) arm_poll_backend(slot, POLLOUT);
					else         arm_poll_client(slot, POLLOUT);
					return 0;
				}
				// pipe drained; try to pull more (only if src still open).
				if (*state == DIR_OPEN) continue;
				// src EOFed and pipe drained — finish direction.
				shutdown(dst, SHUT_WR);
				*state = DIR_DEAD;
				return 0;
			}
			if (n == 0) {
				// shouldn't happen for a non-empty pipe; treat as error.
				return -1;
			}
			if (errno == EAGAIN || errno == EWOULDBLOCK) {
				if (dir_c2b) arm_poll_backend(slot, POLLOUT);
				else         arm_poll_client(slot, POLLOUT);
				return 0;
			}
			if (errno == EINTR) continue;
			return -1;
		}

		// pipe is empty — either we're idle waiting for src, or src EOFed.
		if (*state == DIR_SRC_EOF) {
			shutdown(dst, SHUT_WR);
			*state = DIR_DEAD;
			return 0;
		}
		// Need POLLIN on src to wake us up.
		if (dir_c2b) arm_poll_client(slot, POLLIN);
		else         arm_poll_backend(slot, POLLIN);
		return 0;
	}
}

// Run both directions and tear the conn down if both are dead.
static void run_conn(int slot) {
	conn_t *c = &conns[slot];
	if (drive_dir(slot, 1) < 0 || drive_dir(slot, 0) < 0) {
		teardown_conn(slot);
		return;
	}
	if (c->c2b_state == DIR_DEAD && c->b2c_state == DIR_DEAD) {
		teardown_conn(slot);
	}
}

// ── poll arming ────────────────────────────────────────────────────────────

static void arm_poll_client(int slot, unsigned mask) {
	conn_t *c = &conns[slot];
	if (c->poll_c_armed && c->poll_c_mask == mask) return;
	struct io_uring_sqe *sqe = ring_get_sqe(&R);
	if (!sqe) return;  // ring full; conn will idle until next iteration. Acceptable: caller is in CQE loop, we'll drain soon.
	prep_poll_add(sqe, c->fd_client, mask,
	              mk_ud((uint16_t)slot, OP_POLL_C, c->gen));
	c->poll_c_armed = 1;
	c->poll_c_mask  = mask;
}

static void arm_poll_backend(int slot, unsigned mask) {
	conn_t *c = &conns[slot];
	if (c->poll_b_armed && c->poll_b_mask == mask) return;
	struct io_uring_sqe *sqe = ring_get_sqe(&R);
	if (!sqe) return;
	prep_poll_add(sqe, c->fd_backend, mask,
	              mk_ud((uint16_t)slot, OP_POLL_B, c->gen));
	c->poll_b_armed = 1;
	c->poll_b_mask  = mask;
}

// ── connect to backend ─────────────────────────────────────────────────────

static int submit_connect(int slot) {
	conn_t *c = &conns[slot];

	int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
	if (fd < 0) return -1;
	c->fd_backend = fd;

	int idx = (int)((rr_counter++) % (uint64_t)n_backends);
	struct sockaddr_un *a = &backend_addrs[idx];

	struct io_uring_sqe *sqe = ring_get_sqe(&R);
	if (!sqe) {
		close(fd);
		c->fd_backend = -1;
		return -1;
	}
	prep_connect(sqe, fd, (struct sockaddr *)a,
	             (socklen_t)(offsetof(struct sockaddr_un, sun_path) + strlen(a->sun_path) + 1),
	             mk_ud((uint16_t)slot, OP_CONNECT, c->gen));
	return 0;
}

// ── handle CQEs ────────────────────────────────────────────────────────────

static int listen_fd_g;

static void arm_accept(void) {
	struct io_uring_sqe *sqe = ring_get_sqe(&R);
	if (!sqe) return;
	prep_accept_multishot(sqe, listen_fd_g, mk_ud(0, OP_ACCEPT, 0));
}

static void on_accept(struct io_uring_cqe *cqe) {
	int more = (cqe->flags & IORING_CQE_F_MORE) != 0;
	if (cqe->res >= 0) {
		int client_fd = cqe->res;
		// Optional TCP_NODELAY: small bodies, latency-sensitive.
		int one = 1;
		setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
		set_nonblock(client_fd);

		int slot = alloc_slot();
		if (slot < 0) { close(client_fd); goto done; }

		conn_t *c = &conns[slot];
		c->in_use      = 1;
		c->fd_client   = client_fd;
		c->fd_backend  = -1;
		c->pipe_c2b[0] = c->pipe_c2b[1] = -1;
		c->pipe_b2c[0] = c->pipe_b2c[1] = -1;
		c->c2b_state   = DIR_DEAD;  // gated until connect completes
		c->b2c_state   = DIR_DEAD;
		c->c2b_in_pipe = 0;
		c->b2c_in_pipe = 0;
		c->poll_c_armed = 0;
		c->poll_b_armed = 0;

		if (submit_connect(slot) < 0) {
			teardown_conn(slot);
		}
	}
done:
	if (!more) arm_accept();
}

static void on_connect(struct io_uring_cqe *cqe) {
	int slot = ud_slot(cqe->user_data);
	conn_t *c = &conns[slot];
	if (!c->in_use || ud_gen(cqe->user_data) != c->gen) return;

	if (cqe->res < 0) {
		teardown_conn(slot);
		return;
	}

	if (pipe2(c->pipe_c2b, O_CLOEXEC | O_NONBLOCK) < 0 ||
	    pipe2(c->pipe_b2c, O_CLOEXEC | O_NONBLOCK) < 0) {
		teardown_conn(slot);
		return;
	}
	fcntl(c->pipe_c2b[0], F_SETPIPE_SZ, PIPE_SZ);
	fcntl(c->pipe_b2c[0], F_SETPIPE_SZ, PIPE_SZ);

	c->c2b_state = DIR_OPEN;
	c->b2c_state = DIR_OPEN;
	run_conn(slot);
}

static void on_poll(struct io_uring_cqe *cqe, int is_client) {
	int slot = ud_slot(cqe->user_data);
	conn_t *c = &conns[slot];
	if (!c->in_use || ud_gen(cqe->user_data) != c->gen) return;

	if (is_client) c->poll_c_armed = 0;
	else           c->poll_b_armed = 0;

	if (cqe->res < 0) {
		teardown_conn(slot);
		return;
	}

	// POLLERR / POLLHUP signal that the fd is broken; drive_dir will detect
	// EOF or error via splice and clean up.
	run_conn(slot);
}

// ── main loop ──────────────────────────────────────────────────────────────

static int parse_listen(const char *spec, struct sockaddr_in *out) {
	// "0.0.0.0:9999" → sockaddr_in
	const char *colon = strchr(spec, ':');
	if (!colon) return -1;
	char host[64];
	size_t hl = (size_t)(colon - spec);
	if (hl >= sizeof host) return -1;
	memcpy(host, spec, hl);
	host[hl] = 0;
	int port = atoi(colon + 1);
	if (port <= 0 || port > 65535) return -1;

	memset(out, 0, sizeof *out);
	out->sin_family = AF_INET;
	out->sin_port   = htons((uint16_t)port);
	if (inet_pton(AF_INET, host, &out->sin_addr) != 1) return -1;
	return 0;
}

static int parse_backends(const char *csv) {
	const char *p = csv;
	n_backends = 0;
	while (*p && n_backends < MAX_BACKENDS) {
		const char *comma = strchr(p, ',');
		size_t l = comma ? (size_t)(comma - p) : strlen(p);
		if (l == 0) break;
		struct sockaddr_un *a = &backend_addrs[n_backends];
		memset(a, 0, sizeof *a);
		a->sun_family = AF_UNIX;
		if (l >= sizeof a->sun_path) return -1;
		memcpy(a->sun_path, p, l);
		a->sun_path[l] = 0;
		n_backends++;
		if (!comma) break;
		p = comma + 1;
	}
	return n_backends > 0 ? 0 : -1;
}

static int open_listen(const struct sockaddr_in *addr) {
	int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (fd < 0) return -1;
	int one = 1;
	setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
	setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof one);
	if (bind(fd, (const struct sockaddr *)addr, sizeof *addr) < 0) { close(fd); return -1; }
	if (listen(fd, LISTEN_BACKLOG) < 0) { close(fd); return -1; }
	return fd;
}

int main(void) {
	signal(SIGPIPE, SIG_IGN);

	const char *listen_spec = getenv("RINHALB_LISTEN");
	if (!listen_spec) listen_spec = "0.0.0.0:9999";
	const char *backends_spec = getenv("RINHALB_BACKENDS");
	if (!backends_spec) {
		fprintf(stderr, "RINHALB_BACKENDS is required (comma-separated UDS paths)\n");
		return 2;
	}

	struct sockaddr_in laddr;
	if (parse_listen(listen_spec, &laddr) < 0) {
		fprintf(stderr, "bad RINHALB_LISTEN: %s\n", listen_spec);
		return 2;
	}
	if (parse_backends(backends_spec) < 0) {
		fprintf(stderr, "bad RINHALB_BACKENDS: %s\n", backends_spec);
		return 2;
	}

	int listen_fd = open_listen(&laddr);
	if (listen_fd < 0) {
		perror("listen");
		return 1;
	}
	listen_fd_g = listen_fd;

	if (ring_init(&R, SQ_DEPTH) < 0) {
		perror("io_uring_setup");
		return 1;
	}

	for (int i = 0; i < MAX_CONNS; i++) {
		conns[i].fd_client = -1;
		conns[i].fd_backend = -1;
		conns[i].pipe_c2b[0] = conns[i].pipe_c2b[1] = -1;
		conns[i].pipe_b2c[0] = conns[i].pipe_b2c[1] = -1;
		conns[i].in_use = 0;
		conns[i].gen = 0;
		free_stack[free_top++] = MAX_CONNS - 1 - i;  // higher slots first
	}

	arm_accept();

	for (;;) {
		int rc = ring_submit_and_wait(&R, 1);
		if (rc < 0 && errno != EINTR && errno != EBUSY) {
			perror("io_uring_enter");
			return 1;
		}

		while (1) {
			unsigned ready = ring_cq_ready(&R);
			if (ready == 0) break;
			if (ready > CQE_BATCH) ready = CQE_BATCH;
			for (unsigned i = 0; i < ready; i++) {
				struct io_uring_cqe *cqe = ring_cq_peek(&R, i);
				switch (ud_op(cqe->user_data)) {
				case OP_ACCEPT:  on_accept(cqe);  break;
				case OP_CONNECT: on_connect(cqe); break;
				case OP_POLL_C:  on_poll(cqe, 1); break;
				case OP_POLL_B:  on_poll(cqe, 0); break;
				default: break;
				}
			}
			ring_cq_advance(&R, ready);
		}
	}
}

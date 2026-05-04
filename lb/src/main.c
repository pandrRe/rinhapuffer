// rinhalb — fd-passing dispatcher.
//
// lb accepts TCP connections on the public port and hands the accepted fd
// to one of the api processes via SCM_RIGHTS over a persistent Unix STREAM
// socket. After the handoff, lb drops its ref and is OUT of the data path:
// the api process recv/sends directly to the client TCP socket.
//
// The kernel keeps the TCP socket alive across the SCM_RIGHTS pass (the
// underlying `struct file *` is refcounted; the socket's `sk_net` is fixed
// at creation in lb's namespace and remains usable from the api's process
// regardless of net-ns). Per-request lb work is zero — only one handoff
// per connection, amortised across many keep-alive requests.
//
// Backend selection: round-robin across N persistent UDS conns (one per
// api process at /sockets/apiN.sock). Connection imbalance is bounded by
// keep-alive distribution.

#define _GNU_SOURCE
#include "ring.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

// ── config ─────────────────────────────────────────────────────────────────

#define MAX_BACKENDS         8
#define LISTEN_BACKLOG       4096
#define SQ_DEPTH             4096
#define CQE_BATCH            256
#define MAX_INFLIGHT_HANDOFF 256

// ── user_data layout ───────────────────────────────────────────────────────
//   bits 0..15  handoff slot index (for SENDMSG completions)
//   bits 16..23 op tag
//   bits 24..31 backend index (for SENDMSG — which UDS conn was used)
//   bits 32..63 client fd (so we can close it on completion without
//                          re-reading any per-handoff state)
//
// ACCEPT carries op tag only; SENDMSG carries op + backend_idx + handoff_idx + client_fd.

enum {
	OP_ACCEPT  = 0,
	OP_SENDMSG = 1,
};

static inline uint64_t mk_ud(uint8_t op, uint8_t backend_idx,
                             uint16_t handoff_idx, int32_t client_fd) {
	return ((uint64_t)(uint32_t)client_fd << 32) |
	       ((uint64_t)backend_idx << 24) |
	       ((uint64_t)op << 16) |
	       (uint64_t)handoff_idx;
}
static inline uint16_t ud_handoff(uint64_t ud) { return (uint16_t)(ud & 0xffff); }
static inline uint8_t  ud_op(uint64_t ud)      { return (uint8_t)((ud >> 16) & 0xff); }
static inline uint8_t  ud_backend(uint64_t ud) { return (uint8_t)((ud >> 24) & 0xff); }
static inline int32_t  ud_fd(uint64_t ud)      { return (int32_t)(ud >> 32); }

// ── per-handoff frame ──────────────────────────────────────────────────────
//
// SENDMSG's msghdr (and its msg_iov + msg_control buffers) MUST outlive the
// SQE submission — the kernel reads them asynchronously. Pool of frames
// keyed by handoff_idx; allocated on accept, freed on sendmsg completion.

typedef struct {
	struct msghdr hdr;
	struct iovec  iov;
	char          cmsg_buf[CMSG_SPACE(sizeof(int))];
	char          payload;        // 1-byte "F" — SCM_RIGHTS over Linux works
	                              // with empty payloads but the convention is
	                              // to send at least 1 byte for portability.
	uint8_t       in_use;
} handoff_t;

static handoff_t handoffs[MAX_INFLIGHT_HANDOFF];
static int       handoff_free_stack[MAX_INFLIGHT_HANDOFF];
static int       handoff_free_top = 0;

static int alloc_handoff(void) {
	if (handoff_free_top == 0) return -1;
	return handoff_free_stack[--handoff_free_top];
}

static void return_handoff(int idx) {
	handoffs[idx].in_use = 0;
	handoff_free_stack[handoff_free_top++] = idx;
}

// ── backends ───────────────────────────────────────────────────────────────

static struct sockaddr_un backend_addrs[MAX_BACKENDS];
static int                backend_fds[MAX_BACKENDS];
static int                n_backends = 0;
static uint64_t           rr_counter = 0;

// ── io_uring + listen state ────────────────────────────────────────────────

static struct ring R;
static int         listen_fd_g;

// ── helpers ────────────────────────────────────────────────────────────────

static socklen_t un_addrlen(const struct sockaddr_un *a) {
	return (socklen_t)(offsetof(struct sockaddr_un, sun_path) +
	                   strlen(a->sun_path) + 1);
}

// Connect to an api UDS at startup, retrying with backoff because the api
// container may not have bound its socket yet (compose `depends_on` only
// waits for container start, not for socket-ready).
static int connect_backend_with_retry(const struct sockaddr_un *addr) {
	const int max_attempts = 40;            // 40 × 250 ms = 10 s
	const struct timespec delay = { .tv_sec = 0, .tv_nsec = 250L * 1000 * 1000 };

	for (int i = 0; i < max_attempts; i++) {
		int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
		if (fd < 0) return -1;
		if (connect(fd, (const struct sockaddr *)addr, un_addrlen(addr)) == 0) {
			return fd;
		}
		int err = errno;
		close(fd);
		if (err != ECONNREFUSED && err != ENOENT && err != EAGAIN) {
			fprintf(stderr, "connect(%s) failed: %s\n",
			        addr->sun_path, strerror(err));
			return -1;
		}
		nanosleep(&delay, NULL);
	}
	fprintf(stderr, "connect(%s) timed out after %d attempts\n",
	        addr->sun_path, max_attempts);
	return -1;
}

// On a sendmsg failure to a backend, drop the broken fd and reconnect. If
// reconnect fails the slot stays with fd<0; subsequent handoffs to that
// slot will fail and trigger another reconnect attempt.
static void reconnect_backend(int idx) {
	if (backend_fds[idx] >= 0) close(backend_fds[idx]);
	backend_fds[idx] = -1;
	int fd = connect_backend_with_retry(&backend_addrs[idx]);
	if (fd < 0) {
		fprintf(stderr, "reconnect to backend %d (%s) failed\n",
		        idx, backend_addrs[idx].sun_path);
		return;
	}
	backend_fds[idx] = fd;
	fprintf(stderr, "reconnected to backend %d (%s)\n",
	        idx, backend_addrs[idx].sun_path);
}

// Build the SCM_RIGHTS frame for a single fd handoff.
static void build_handoff(handoff_t *h, int client_fd) {
	h->payload = 'F';
	h->iov.iov_base = &h->payload;
	h->iov.iov_len = 1;

	h->hdr.msg_name       = NULL;
	h->hdr.msg_namelen    = 0;
	h->hdr.msg_iov        = &h->iov;
	h->hdr.msg_iovlen     = 1;
	h->hdr.msg_control    = h->cmsg_buf;
	h->hdr.msg_controllen = sizeof h->cmsg_buf;
	h->hdr.msg_flags      = 0;

	struct cmsghdr *c = CMSG_FIRSTHDR(&h->hdr);
	c->cmsg_level = SOL_SOCKET;
	c->cmsg_type  = SCM_RIGHTS;
	c->cmsg_len   = CMSG_LEN(sizeof(int));
	memcpy(CMSG_DATA(c), &client_fd, sizeof(int));
}

// ── accept / sendmsg ───────────────────────────────────────────────────────

static void arm_accept(void) {
	struct io_uring_sqe *sqe = ring_get_sqe(&R);
	if (!sqe) return;
	prep_accept_multishot(sqe, listen_fd_g, mk_ud(OP_ACCEPT, 0, 0, 0));
}

static void on_accept(struct io_uring_cqe *cqe) {
	int more = (cqe->flags & IORING_CQE_F_MORE) != 0;
	if (cqe->res < 0) {
		// Multishot retired (rare). Re-arm.
		if (!more) arm_accept();
		return;
	}
	int client_fd = cqe->res;
	// TCP_NODELAY: small JSON bodies, latency-sensitive eval.
	int one = 1;
	setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);

	int handoff_idx = alloc_handoff();
	if (handoff_idx < 0) {
		// Pool exhausted (extremely unlikely at our load). Drop the conn.
		close(client_fd);
		if (!more) arm_accept();
		return;
	}
	int backend_idx = (int)((rr_counter++) % (uint64_t)n_backends);
	if (backend_fds[backend_idx] < 0) {
		// Backend unavailable; try next, or give up if all are down.
		// One quick retry across remaining backends.
		int picked = -1;
		for (int i = 0; i < n_backends; i++) {
			int j = (backend_idx + 1 + i) % n_backends;
			if (backend_fds[j] >= 0) { picked = j; break; }
		}
		if (picked < 0) {
			return_handoff(handoff_idx);
			close(client_fd);
			if (!more) arm_accept();
			return;
		}
		backend_idx = picked;
	}

	handoff_t *h = &handoffs[handoff_idx];
	h->in_use = 1;
	build_handoff(h, client_fd);

	struct io_uring_sqe *sqe = ring_get_sqe(&R);
	if (!sqe) {
		return_handoff(handoff_idx);
		close(client_fd);
		if (!more) arm_accept();
		return;
	}
	prep_sendmsg(sqe, backend_fds[backend_idx], &h->hdr, MSG_NOSIGNAL,
	             mk_ud(OP_SENDMSG, (uint8_t)backend_idx,
	                   (uint16_t)handoff_idx, client_fd));

	// io_uring multishot accept stays armed automatically. Re-arm only
	// if the kernel signalled retirement via !F_MORE.
	if (!more) arm_accept();
}

static void on_sendmsg(struct io_uring_cqe *cqe) {
	int handoff_idx = ud_handoff(cqe->user_data);
	int backend_idx = ud_backend(cqe->user_data);
	int client_fd   = ud_fd(cqe->user_data);

	if (cqe->res < 0) {
		fprintf(stderr, "sendmsg → backend %d failed: %s\n",
		        backend_idx, strerror(-cqe->res));
		// Drop the client conn; reconnect the backend.
		close(client_fd);
		reconnect_backend(backend_idx);
	} else {
		// Successful handoff — the api now holds a refcounted copy of the
		// kernel socket. Drop our fd; the kernel keeps the socket alive.
		close(client_fd);
	}
	return_handoff(handoff_idx);
}

// ── env parsing + listen socket (mostly unchanged from old main.c) ─────────

static int parse_listen(const char *spec, struct sockaddr_in *out) {
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
	if (bind(fd, (const struct sockaddr *)addr, sizeof *addr) < 0) {
		close(fd);
		return -1;
	}
	if (listen(fd, LISTEN_BACKLOG) < 0) {
		close(fd);
		return -1;
	}
	return fd;
}

// ── main ───────────────────────────────────────────────────────────────────

int main(void) {
	signal(SIGPIPE, SIG_IGN);

	const char *listen_spec = getenv("RINHALB_LISTEN");
	if (!listen_spec) listen_spec = "0.0.0.0:9999";
	const char *backends_spec = getenv("RINHALB_BACKENDS");
	if (!backends_spec) {
		fprintf(stderr,
		        "RINHALB_BACKENDS is required (comma-separated UDS paths)\n");
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

	// Connect to all backends BEFORE opening the listen socket; this way
	// clients aren't accepted until we have somewhere to hand them off.
	for (int i = 0; i < n_backends; i++) backend_fds[i] = -1;
	for (int i = 0; i < n_backends; i++) {
		int fd = connect_backend_with_retry(&backend_addrs[i]);
		if (fd < 0) {
			fprintf(stderr, "failed to connect to backend %d (%s)\n",
			        i, backend_addrs[i].sun_path);
			return 1;
		}
		backend_fds[i] = fd;
		fprintf(stderr, "connected to backend %d (%s)\n",
		        i, backend_addrs[i].sun_path);
	}

	listen_fd_g = open_listen(&laddr);
	if (listen_fd_g < 0) {
		perror("listen");
		return 1;
	}

	if (ring_init(&R, SQ_DEPTH) < 0) {
		perror("io_uring_setup");
		return 1;
	}

	for (int i = 0; i < MAX_INFLIGHT_HANDOFF; i++) {
		handoff_free_stack[handoff_free_top++] = MAX_INFLIGHT_HANDOFF - 1 - i;
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
				case OP_SENDMSG: on_sendmsg(cqe); break;
				default: break;
				}
			}
			ring_cq_advance(&R, ready);
		}
	}
}

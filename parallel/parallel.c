/* parallel.c — Parallel n x n matrix multiplication (C = A x B) with OpenMP */
#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#define MAX_MEM_SHARE 0.5

typedef enum { ALGO_IJK, ALGO_IKJ, ALGO_BLOCKED } algo_t;
static const char *ALGO_NAMES[] = {"ijk", "ikj", "blocked"};

typedef struct {
    int n, threads, block, reps, samples, hold;
    unsigned seed;
    algo_t algo;
    int csv, header;
} options_t;

/* ------------------------------------------------------------------ helpers */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -n N        matrix size n (default 800)\n"
        "  -t T        number of OpenMP threads (default: all logical CPUs)\n"
        "  -a ALGO     ijk | ikj | blocked (default ikj)\n"
        "  -b B        block (tile) size for 'blocked' (default 64)\n"
        "  -r R        repetitions of the multiply phase (default 3)\n"
        "  -k K        number of random entries to verify (default 64)\n"
        "  -s SEED     seed for the matrix values (default 42)\n"
        "  --hold S    sleep S seconds after multiplying, before freeing memory\n"
        "              (lets the task-2 memory-map tool inspect this process)\n"
        "  --csv       print one CSV line instead of a human-readable report\n"
        "  --header    print the CSV header line and exit\n"
        "Loop schedule comes from OMP_SCHEDULE (e.g. static, dynamic,4, guided);\n"
        "if unset this program uses 'static'.\n", prog);
}

static const char *CSV_HEADER =
    "n,threads,algo,block,schedule,chunk,reps,t_init,t_mult_med,t_mult_min,t_mult_max,"
    "t_verify,t_total,gflops,maxrss_kb,vmpeak_kb,checksum,ok";

static long read_proc_kb(const char *path, const char *key) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    long value = -1;
    size_t keylen = strlen(key);
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, keylen) == 0 && line[keylen] == ':') {
            value = strtol(line + keylen + 1, NULL, 10);
            break;
        }
    }
    fclose(f);
    return value;
}

static inline double next_value(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return (double)x / 4294967296.0;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static inline int imin(int a, int b) { return a < b ? a : b; }

/* ------------------------------------------------------------- the kernels */

static void matmul_ijk(int n, const double *restrict A, const double *restrict B,
                       double *restrict C) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

static void matmul_ikj(int n, const double *restrict A, const double *restrict B,
                       double *restrict C) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[k * n + j];
        }
}

static void matmul_blocked(int n, int bs, const double *restrict A,
                           const double *restrict B, double *restrict C) {
    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int ii = 0; ii < n; ii += bs)
        for (int jj = 0; jj < n; jj += bs)
            for (int kk = 0; kk < n; kk += bs) {
                int imax = imin(ii + bs, n), jmax = imin(jj + bs, n), kmax = imin(kk + bs, n);
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        double a = A[i * n + k];
                        for (int j = jj; j < jmax; j++)
                            C[i * n + j] += a * B[k * n + j];
                    }
            }
}

/* ------------------------------------------------------------- arguments */

static int parse_args(int argc, char **argv, options_t *o) {
    *o = (options_t){.n = 800, .threads = omp_get_num_procs(), .block = 64, .reps = 3,
                     .samples = 64, .seed = 42, .algo = ALGO_IKJ};
    static struct option longopts[] = {
        {"hold", required_argument, NULL, 'H'}, {"csv", no_argument, NULL, 'C'},
        {"header", no_argument, NULL, 'D'},     {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}};
    int c;
    while ((c = getopt_long(argc, argv, "n:t:a:b:r:k:s:h", longopts, NULL)) != -1) {
        switch (c) {
        case 'n': o->n = atoi(optarg); break;
        case 't': o->threads = atoi(optarg); break;
        case 'b': o->block = atoi(optarg); break;
        case 'r': o->reps = atoi(optarg); break;
        case 'k': o->samples = atoi(optarg); break;
        case 's': o->seed = (unsigned)strtoul(optarg, NULL, 10); break;
        case 'H': o->hold = atoi(optarg); break;
        case 'C': o->csv = 1; break;
        case 'D': o->header = 1; break;
        case 'a':
            if (!strcmp(optarg, "ijk")) o->algo = ALGO_IJK;
            else if (!strcmp(optarg, "ikj")) o->algo = ALGO_IKJ;
            else if (!strcmp(optarg, "blocked")) o->algo = ALGO_BLOCKED;
            else { fprintf(stderr, "unknown algorithm '%s'\n", optarg); return -1; }
            break;
        case 'h': usage(argv[0]); exit(0);
        default: usage(argv[0]); return -1;
        }
    }
    int cpus = omp_get_num_procs();
    if (o->n < 1 || o->n > 20000) { fprintf(stderr, "n must be in 1..20000\n"); return -1; }
    if (o->threads < 1 || o->threads > cpus) {
        fprintf(stderr, "threads must be in 1..%d (logical CPUs on this machine)\n", cpus);
        return -1;
    }
    if (o->block < 1 || o->reps < 1 || o->reps > 100 || o->samples < 0 || o->hold < 0 ||
        o->hold > 600) {
        fprintf(stderr, "invalid -b/-r/-k/--hold value (reps 1..100, hold 0..600 s)\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    options_t o;
    if (parse_args(argc, argv, &o) != 0) return 2;
    if (o.header) { puts(CSV_HEADER); return 0; }

    const int n = o.n;
    const size_t bytes = (size_t)n * n * sizeof(double);

    long avail_kb = read_proc_kb("/proc/meminfo", "MemAvailable");
    if (avail_kb > 0 && 3.0 * bytes > MAX_MEM_SHARE * avail_kb * 1024.0) {
        fprintf(stderr, "refusing: 3 x %.1f MB needed but only %.1f MB available "
                "(limit %.0f%%)\n", bytes / 1e6, avail_kb / 1e3, MAX_MEM_SHARE * 100);
        return 3;
    }

    omp_set_num_threads(o.threads);
    if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_static, 0);
    omp_sched_t sched_kind;
    int chunk;
    omp_get_schedule(&sched_kind, &chunk);
    const char *sched_name = sched_kind == omp_sched_static  ? "static"
                           : sched_kind == omp_sched_dynamic ? "dynamic"
                           : sched_kind == omp_sched_guided  ? "guided" : "auto";

    double t_start = omp_get_wtime();

    /* ---- phase 1: init (serial) */
    double *A = malloc(bytes), *B = malloc(bytes), *C = malloc(bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed: %s\n", strerror(errno));
        free(A); free(B); free(C);
        return 3;
    }
    uint32_t state = o.seed ? o.seed : 1;
    for (size_t i = 0; i < (size_t)n * n; i++) A[i] = next_value(&state);
    for (size_t i = 0; i < (size_t)n * n; i++) B[i] = next_value(&state);
    memset(C, 0, bytes);
    double t_init = omp_get_wtime() - t_start;

    /* ---- phase 2: multiply (parallel), repeated */
    double *times = malloc(sizeof(double) * o.reps);
    if (!times) { free(A); free(B); free(C); return 3; }
    for (int r = 0; r < o.reps; r++) {
        if (o.algo != ALGO_IJK) memset(C, 0, bytes);
        double t0 = omp_get_wtime();
        switch (o.algo) {
        case ALGO_IJK: matmul_ijk(n, A, B, C); break;
        case ALGO_IKJ: matmul_ikj(n, A, B, C); break;
        case ALGO_BLOCKED: matmul_blocked(n, o.block, A, B, C); break;
        }
        times[r] = omp_get_wtime() - t0;
    }
    qsort(times, o.reps, sizeof(double), cmp_double);
    double t_med = o.reps % 2 ? times[o.reps / 2]
                              : 0.5 * (times[o.reps / 2 - 1] + times[o.reps / 2]);
    double t_min = times[0], t_max = times[o.reps - 1];
    free(times);

    if (o.hold > 0) {
        fprintf(stderr, "[parallel] pid %d holding for %d s (inspect with: python3 mmap.py %d)\n",
                (int)getpid(), o.hold, (int)getpid());
        sleep((unsigned)o.hold);
    }

    /* ---- phase 3: verify (serial) */
    double t_v0 = omp_get_wtime();
    int ok = 1;
    uint32_t vstate = 12345;
    for (int s = 0; s < o.samples; s++) {
        int i = (int)(next_value(&vstate) * n), j = (int)(next_value(&vstate) * n);
        double ref = 0.0;
        for (int k = 0; k < n; k++) ref += A[i * n + k] * B[k * n + j];
        if (fabs(C[i * n + j] - ref) > 1e-9 * fabs(ref)) {
            if (!o.csv) fprintf(stderr, "mismatch at (%d,%d): %.17g vs %.17g\n", i, j,
                                C[i * n + j], ref);
            ok = 0;
        }
    }
    double checksum = 0.0;
    for (size_t i = 0; i < (size_t)n * n; i++) checksum += C[i];
    double t_verify = omp_get_wtime() - t_v0;

    double t_total = t_init + t_med + t_verify;
    double gflops = 2.0 * n * (double)n * n / t_med / 1e9;

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    long vmpeak_kb = read_proc_kb("/proc/self/status", "VmPeak");

    if (o.csv) {
        printf("%d,%d,%s,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%ld,%ld,%.10e,%d\n",
               n, o.threads, ALGO_NAMES[o.algo], o.algo == ALGO_BLOCKED ? o.block : 0,
               sched_name, chunk, o.reps, t_init, t_med, t_min, t_max, t_verify, t_total,
               gflops, ru.ru_maxrss, vmpeak_kb, checksum, ok);
    } else {
        printf("n=%d  threads=%d  algo=%s", n, o.threads, ALGO_NAMES[o.algo]);
        if (o.algo == ALGO_BLOCKED) printf("  block=%d", o.block);
        printf("  schedule=%s,%d\n", sched_name, chunk);
        printf("  init     (serial)   %10.4f s\n", t_init);
        printf("  multiply (parallel) %10.4f s  median of %d  [min %.4f, max %.4f]\n",
               t_med, o.reps, t_min, t_max);
        printf("  verify   (serial)   %10.4f s  (%d samples)\n", t_verify, o.samples);
        printf("  performance         %10.2f GFLOP/s\n", gflops);
        printf("  memory   theory %.1f MB | peak RSS %.1f MB | peak virtual %.1f MB\n",
               3.0 * bytes / 1e6, ru.ru_maxrss / 1e3, vmpeak_kb / 1e3);
        printf("  checksum %.10e  -> %s\n", checksum, ok ? "OK" : "WRONG");
    }

    free(A); free(B); free(C);
    return ok ? 0 : 1;
}

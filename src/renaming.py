"""Identifier renaming with tree-sitter (the renamed test set and the augmented train set).

Owner: Ali Khalid Albinali

Every user-defined function name becomes FUN1, FUN2, ... and every other user-defined
identifier becomes VAR1, VAR2, ..., numbered by first appearance inside each function.
Keywords, library calls (LIBRARY_FUNCTIONS), KEEP_IDENTIFIERS, ALL_CAPS names, struct
fields (field_identifier), type names (type_identifier) and goto labels are not changed,
so the program does the same thing and the label stays valid.

How it works:
  1. Parse the function with tree-sitter-c (on UTF-8 bytes).
  2. Walk the tree and collect every node of type ``identifier``. Struct members are
     ``field_identifier``, type names are ``type_identifier`` and goto labels are
     ``statement_identifier``, so they are never collected.
  3. The name in the function definition's declarator gets FUN1. A name used as the
     ``function`` of a ``call_expression`` gets the next FUNk, every other name the next VARk,
     in order of first appearance. A name keeps its first new name everywhere (also inside
     ERROR nodes).
  4. Replace from the last byte offset to the first, so earlier offsets stay valid.
  5. Parse again and compare the number of ERROR / MISSING nodes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------- name lists

C_KEYWORDS: frozenset[str] = frozenset("""
auto break case char const continue default do double else enum extern float for goto if
inline int long register restrict return short signed sizeof static struct switch typedef
union unsigned void volatile while _Alignas _Alignof _Atomic _Bool _Complex _Generic
_Imaginary _Noreturn _Static_assert _Thread_local alignas alignof bool static_assert
thread_local typeof asm
""".split())

_LIB = """
assert static_assert

isalnum isalpha isascii isblank iscntrl isdigit isgraph islower isprint ispunct isspace
isupper isxdigit tolower toupper toascii

strerror strerror_r perror

acos acosf acosl asin asinf asinl atan atanf atanl atan2 atan2f atan2l cos cosf cosl sin sinf
sinl tan tanf tanl acosh acoshf asinh asinhf atanh atanhf cosh coshf sinh sinhf tanh tanhf
exp expf expl exp2 exp2f expm1 expm1f frexp frexpf ldexp ldexpf log logf logl log10 log10f
log10l log1p log1pf log2 log2f log2l logb ilogb modf modff scalbn scalbnf scalbln cbrt cbrtf
fabs fabsf fabsl hypot hypotf pow powf powl sqrt sqrtf sqrtl erf erff erfc erfcf lgamma
tgamma ceil ceilf ceill floor floorf floorl nearbyint rint rintf lrint lrintf llrint llrintf
round roundf roundl lround lroundf llround llroundf trunc truncf truncl fmod fmodf remainder
remquo copysign copysignf nan nanf nextafter nexttoward fdim fmax fmaxf fmin fminf fma fmaf
isnan isinf isfinite isnormal signbit fpclassify

setjmp longjmp sigsetjmp siglongjmp

signal raise sigaction sigemptyset sigfillset sigaddset sigdelset sigismember sigprocmask
sigsuspend sigwait kill killpg alarm pause

va_start va_arg va_end va_copy

offsetof

fopen freopen fclose fflush fdopen fileno setbuf setvbuf fprintf printf sprintf snprintf
vfprintf vprintf vsprintf vsnprintf dprintf vdprintf asprintf vasprintf fscanf scanf sscanf
vfscanf vscanf vsscanf fgetc fgets fputc fputs getc getchar gets putc putchar puts ungetc
fread fwrite fgetpos fseek fseeko fsetpos ftell ftello rewind clearerr feof ferror remove
rename tmpfile tmpnam tempnam getline getdelim popen pclose flockfile funlockfile
getc_unlocked putc_unlocked fmemopen open_memstream

atof atoi atol atoll strtod strtof strtold strtol strtoll strtoul strtoull rand srand random
srandom rand_r drand48 lrand48 mrand48 srand48 calloc free malloc realloc reallocarray
aligned_alloc posix_memalign memalign valloc abort atexit at_quick_exit exit _Exit _exit
quick_exit getenv setenv unsetenv putenv secure_getenv system bsearch qsort abs labs llabs
div ldiv lldiv mblen mbtowc wctomb mbstowcs wcstombs mkstemp mkdtemp mktemp realpath

memchr memcmp memcpy memmove memset memrchr memmem mempcpy memccpy strcat strncat strchr
strrchr strcmp strncmp strcoll strcpy strncpy stpcpy stpncpy strcspn strlen strnlen strpbrk
strspn strstr strcasestr strtok strtok_r strxfrm strdup strndup strsep strlcpy strlcat
strcasecmp strncasecmp strsignal bzero bcopy bcmp explicit_bzero index rindex ffs ffsl ffsll

wcslen wcscpy wcsncpy wcscat wcscmp wcsncmp wcschr wcsrchr wcsstr wcstol wcstoul wcstod
wmemcpy wmemmove wmemset wmemcmp wmemchr mbrtowc wcrtomb mbsrtowcs wcsrtombs btowc wctob
towlower towupper iswalpha iswdigit iswspace iswupper iswlower iswalnum iswprint swprintf
fwprintf wprintf

clock time difftime mktime asctime ctime gmtime localtime gmtime_r localtime_r strftime
strptime timegm nanosleep clock_gettime clock_settime clock_getres gettimeofday settimeofday
usleep sleep times

setlocale localeconv

atomic_init atomic_load atomic_store atomic_exchange atomic_fetch_add atomic_fetch_sub
atomic_fetch_and atomic_fetch_or atomic_fetch_xor atomic_compare_exchange_strong
atomic_compare_exchange_weak atomic_thread_fence atomic_load_explicit atomic_store_explicit
atomic_fetch_add_explicit atomic_fetch_sub_explicit

open openat creat close read write pread pwrite readv writev preadv pwritev lseek lseek64
dup dup2 dup3 pipe pipe2 fcntl ioctl fsync fdatasync ftruncate truncate unlink unlinkat
rmdir mkdir mkdirat chdir fchdir getcwd chmod fchmod chown fchown lchown access faccessat
link symlink readlink readlinkat stat fstat lstat fstatat statvfs fstatvfs umask isatty
ttyname sync syncfs posix_fallocate fallocate posix_fadvise sendfile splice tee

mmap munmap mprotect msync madvise posix_madvise mlock munlock mlockall munlockall mremap
shm_open shm_unlink

opendir fdopendir readdir readdir_r closedir rewinddir seekdir telldir scandir dirfd

fork vfork execl execle execlp execv execve execvp execvpe wait waitpid waitid getpid
getppid getpgid setpgid getsid setsid getuid geteuid getgid getegid setuid seteuid setgid
setegid getgroups setgroups nice getpriority setpriority getrlimit setrlimit getrusage
sysconf pathconf fpathconf gethostname sethostname uname prctl syscall daemon chroot

select pselect poll ppoll epoll_create epoll_create1 epoll_ctl epoll_wait eventfd
timerfd_create timerfd_settime timerfd_gettime signalfd inotify_init inotify_add_watch

socket socketpair bind listen accept accept4 connect send sendto sendmsg recv recvfrom
recvmsg shutdown getsockopt setsockopt getsockname getpeername getaddrinfo freeaddrinfo
gai_strerror getnameinfo gethostbyname gethostbyaddr inet_addr inet_aton inet_ntoa inet_ntop
inet_pton htons htonl ntohs ntohl

pthread_create pthread_join pthread_detach pthread_exit pthread_self pthread_equal
pthread_cancel pthread_kill pthread_once pthread_attr_init pthread_attr_destroy
pthread_attr_setdetachstate pthread_attr_setstacksize pthread_attr_getstacksize
pthread_mutex_init pthread_mutex_destroy pthread_mutex_lock pthread_mutex_trylock
pthread_mutex_unlock pthread_mutex_timedlock pthread_mutexattr_init pthread_mutexattr_destroy
pthread_mutexattr_settype pthread_cond_init pthread_cond_destroy pthread_cond_wait
pthread_cond_timedwait pthread_cond_signal pthread_cond_broadcast pthread_condattr_init
pthread_condattr_destroy pthread_rwlock_init pthread_rwlock_destroy pthread_rwlock_rdlock
pthread_rwlock_wrlock pthread_rwlock_unlock pthread_rwlock_tryrdlock pthread_rwlock_trywrlock
pthread_key_create pthread_key_delete pthread_getspecific pthread_setspecific
pthread_spin_init pthread_spin_destroy pthread_spin_lock pthread_spin_unlock
pthread_spin_trylock pthread_barrier_init pthread_barrier_destroy pthread_barrier_wait
pthread_sigmask pthread_setname_np pthread_getname_np pthread_setaffinity_np
pthread_getaffinity_np pthread_setschedparam pthread_getschedparam sched_yield
sched_setaffinity sched_getaffinity sem_init sem_destroy sem_wait sem_trywait sem_timedwait
sem_post sem_getvalue sem_open sem_close sem_unlink

dlopen dlsym dlclose dlerror

getopt getopt_long getopt_long_only

regcomp regexec regerror regfree

iconv iconv_open iconv_close

openlog syslog closelog vsyslog

__builtin_expect __builtin_memcpy __builtin_memset __builtin_strlen __builtin_clz
__builtin_clzll __builtin_ctz __builtin_ctzll __builtin_popcount __builtin_popcountll
__builtin_bswap16 __builtin_bswap32 __builtin_bswap64 __builtin_unreachable __builtin_trap
__builtin_constant_p __builtin_offsetof __builtin_prefetch __builtin_return_address
__builtin_frame_address __builtin_add_overflow __builtin_sub_overflow __builtin_mul_overflow
__builtin_types_compatible_p __builtin_choose_expr __builtin_va_start __builtin_va_end
__builtin_va_arg __builtin_va_copy __builtin_alloca alloca
"""

# C standard library + common POSIX names that are kept as-is when called.
LIBRARY_FUNCTIONS: frozenset[str] = frozenset(_LIB.split())

_KEEP = """
NULL true false errno stdin stdout stderr EOF environ optarg optind opterr optopt
__func__ __FUNCTION__ __PRETTY_FUNCTION__ __FILE__ __LINE__ __DATE__ __TIME__
size_t ssize_t ptrdiff_t intptr_t uintptr_t intmax_t uintmax_t off_t off64_t
int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t
int_least8_t int_least16_t int_least32_t int_least64_t uint_least8_t uint_least16_t
uint_least32_t uint_least64_t int_fast8_t int_fast16_t int_fast32_t int_fast64_t
uint_fast8_t uint_fast16_t uint_fast32_t uint_fast64_t
wchar_t wint_t char16_t char32_t bool va_list FILE fpos_t time_t clock_t suseconds_t
pid_t uid_t gid_t mode_t dev_t ino_t nlink_t blksize_t blkcnt_t socklen_t sa_family_t
in_addr_t in_port_t pthread_t pthread_mutex_t pthread_cond_t pthread_attr_t
pthread_rwlock_t pthread_key_t pthread_once_t sem_t sig_atomic_t sigset_t jmp_buf
sigjmp_buf max_align_t nullptr_t
"""

# Names that are never renamed (NULL, true, false, errno, stdin, stdout, stderr, EOF, ...).
# Standard type names are listed too: inside macro arguments (e.g. DECLARE_ALIGNED(16, uint8_t, buf))
# tree-sitter sometimes parses them as plain identifiers.
KEEP_IDENTIFIERS: frozenset[str] = frozenset(_KEEP.split())

_HAS_LOWER = re.compile(r"[a-z]")


def _is_all_caps(name: str) -> bool:
    """Macros and constants such as AV_LOG_ERROR, EINVAL, FFMIN (at least one letter, no lowercase)."""
    return any(c.isalpha() for c in name) and not _HAS_LOWER.search(name)


def _never_rename(name: str) -> bool:
    return (name in C_KEYWORDS or name in LIBRARY_FUNCTIONS or name in KEEP_IDENTIFIERS
            or _is_all_caps(name) or name.startswith("__")       # __attribute__, __builtin_*, ...
            or re.fullmatch(r"(VAR|FUN)\d+", name) is not None)  # already renamed


# ----------------------------------------------------------------------------- parser

_PARSER = None


def _parser():
    """One shared tree-sitter parser for C (works with tree-sitter 0.22+)."""
    global _PARSER
    if _PARSER is None:
        import tree_sitter as ts
        import tree_sitter_c as tsc

        lang = ts.Language(tsc.language())
        try:
            _PARSER = ts.Parser(lang)
        except TypeError:          # older API
            _PARSER = ts.Parser()
            _PARSER.set_language(lang)
    return _PARSER


def _walk(node):
    """Iterative pre-order walk (deep functions would hit Python's recursion limit)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _count_errors(root) -> int:
    """Number of ERROR nodes plus MISSING nodes in the tree."""
    return sum(1 for n in _walk(root) if n.type == "ERROR" or n.is_missing)


def _definition_name_node(root):
    """The identifier node of the function's own name in its definition, or None.

    Follows the ``declarator`` fields from the first function_definition:
    function_definition -> (pointer_declarator ->)* function_declarator -> identifier.
    """
    func_def = next((n for n in _walk(root) if n.type == "function_definition"), None)
    if func_def is None:
        return None
    node = func_def.child_by_field_name("declarator")
    for _ in range(20):
        if node is None:
            return None
        if node.type == "identifier":
            return node
        if node.type == "function_declarator":
            node = node.child_by_field_name("declarator")
            continue
        # pointer_declarator, parenthesized_declarator, attributed_declarator, ...
        inner = node.child_by_field_name("declarator")
        if inner is None:
            inner = next((c for c in node.named_children
                          if c.type in ("identifier", "function_declarator", "pointer_declarator",
                                        "parenthesized_declarator")), None)
        node = inner
    return None


# Parents of an identifier that declares a variable or parameter.
_DECLARATOR_PARENTS = frozenset({"parameter_declaration", "declaration", "init_declarator",
                                 "pointer_declarator", "array_declarator", "parenthesized_declarator"})


def _is_call_target(node) -> bool:
    parent = node.parent
    return (parent is not None and parent.type == "call_expression"
            and parent.child_by_field_name("function") == node)


# ----------------------------------------------------------------------------- renaming

def rename_identifiers(code: str) -> tuple[str, dict]:
    """Rename user-defined identifiers in one C function.

    Returns:
        (renamed_code, info) with
        ``info = {"changed": bool, "n_var": int, "n_fun": int, "mapping": dict[str, str],
                  "parse_errors_before": int, "parse_errors_after": int}``
    """
    parser = _parser()
    src = code.encode("utf-8")
    tree = parser.parse(src)
    root = tree.root_node
    errors_before = _count_errors(root)

    # Names that tree-sitter sees as a type somewhere in this function are type names:
    # keep them even where they show up as a plain identifier (e.g. sizeof(AVFrame)).
    # A name declared as a variable or parameter is never a type, even if error recovery in a
    # broken parse tags one of its uses as type_identifier; such uses are renamed too (only in
    # trees with parse errors), so the same name always gets the same new name.
    nodes = list(_walk(root))
    declared = {n.text.decode("utf-8", "replace") for n in nodes
                if n.type == "identifier" and n.parent is not None and n.parent.type in _DECLARATOR_PARENTS}
    type_names = {n.text.decode("utf-8", "replace") for n in nodes if n.type == "type_identifier"} - declared

    idents = [n for n in nodes if n.type == "identifier"
              or (errors_before and n.type == "type_identifier"
                  and n.text.decode("utf-8", "replace") in declared)]
    idents.sort(key=lambda n: n.start_byte)

    mapping: dict[str, str] = {}
    n_var = n_fun = 0

    own = _definition_name_node(root)
    if own is not None:
        own_name = own.text.decode("utf-8", "replace")
        if not _never_rename(own_name):
            n_fun = 1
            mapping[own_name] = "FUN1"

    edits: list[tuple[int, int, bytes]] = []
    for node in idents:
        name = node.text.decode("utf-8", "replace")
        if name not in mapping:
            if _never_rename(name) or name in type_names:
                continue
            if _is_call_target(node):
                n_fun += 1
                mapping[name] = f"FUN{n_fun}"
            else:
                n_var += 1
                mapping[name] = f"VAR{n_var}"
        edits.append((node.start_byte, node.end_byte, mapping[name].encode("utf-8")))

    out = src
    for start, end, new in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + new + out[end:]
    renamed = out.decode("utf-8", "replace")

    errors_after = _count_errors(parser.parse(out).root_node) if edits else errors_before

    info = {
        "changed": renamed != code,
        "n_var": n_var,
        "n_fun": n_fun,
        "mapping": mapping,
        "parse_errors_before": errors_before,
        "parse_errors_after": errors_after,
    }
    return renamed, info


def build_renamed_split(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename every function in ``df`` (columns id, code, label).

    Returns:
        (renamed_df with the same ids and labels, stats) with stats shaped like
        ``metrics_summary["renaming"]``::

            {"test_functions_changed_pct", "mean_identifiers_renamed",
             "mean_var_renamed", "mean_fun_renamed", "new_parse_errors"}

        ``test_functions_changed_pct`` is a fraction (0-1), like the other ``pct`` values
        in ``results/``. ``new_parse_errors`` is the number of functions with more parse
        errors after renaming than before (should be 0); their ids are printed.
    """
    codes, infos = [], []
    for code in df["code"].astype(str):
        new_code, info = rename_identifiers(code)
        codes.append(new_code)
        infos.append(info)

    out = df.copy()
    out["code"] = codes
    if "hash" in out.columns:  # the hash belongs to the original code; drop it to avoid confusion
        out = out.drop(columns=["hash"])

    n = max(len(infos), 1)
    worse_ids = [int(i) for i, info in zip(df["id"], infos)
                 if info["parse_errors_after"] > info["parse_errors_before"]]
    if worse_ids:
        print(f"WARNING: {len(worse_ids)} function(s) have more parse errors after renaming: {worse_ids[:20]}")

    stats = {
        "test_functions_changed_pct": round(sum(i["changed"] for i in infos) / n, 4),
        "mean_identifiers_renamed": round(sum(i["n_var"] + i["n_fun"] for i in infos) / n, 2),
        "mean_var_renamed": round(sum(i["n_var"] for i in infos) / n, 2),
        "mean_fun_renamed": round(sum(i["n_fun"] for i in infos) / n, 2),
        "new_parse_errors": len(worse_ids),
    }
    return out.reset_index(drop=True), stats


def build_augmented_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """Original rows + a renamed copy of every row (same labels), with a bool column ``is_renamed``."""
    original = train_df.copy()
    if "hash" in original.columns:
        original = original.drop(columns=["hash"])
    original["is_renamed"] = False

    renamed, _ = build_renamed_split(train_df)
    renamed["is_renamed"] = True

    aug = pd.concat([original, renamed], ignore_index=True)
    aug["is_renamed"] = aug["is_renamed"].astype(bool)
    return aug


# ----------------------------------------------------------------------------- self-test

EXAMPLE = """static int copy_name(char *dst, const char *src, int n) {
    int len = strlen(src);
    if (len > n) len = n;
    memcpy(dst, src, len);
    helper_log(dst);
    return len;
}"""

EXPECTED = ("static int FUN1(char *VAR1, const char *VAR2, int VAR3) { int VAR4 = strlen(VAR2); "
            "if (VAR4 > VAR3) VAR4 = VAR3; memcpy(VAR1, VAR2, VAR4); FUN2(VAR1); return VAR4; }")


def _show(title: str, code: str) -> tuple[str, dict]:
    new_code, info = rename_identifiers(code)
    print(f"=== {title}")
    print("BEFORE:", code)
    print("AFTER: ", new_code)
    print("info:  ", {k: v for k, v in info.items() if k != "mapping"})
    print("map:   ", info["mapping"], "\n")
    return new_code, info


def _short_examples(test_df: pd.DataFrame, k: int, max_len: int = 600) -> pd.DataFrame:
    """k readable test functions (short, changed by renaming, both labels if possible), fixed seed."""
    short = test_df[test_df["code"].str.len().between(150, max_len)]
    pool = short if len(short) >= k else test_df
    picks = []
    for label in (1, 0):
        part = pool[pool["label"] == label]
        picks.append(part.sample(min(len(part), (k + 1) // 2), random_state=404))
    return pd.concat(picks).head(k)


def write_examples_md(test_df: pd.DataFrame, path: str = "results/renaming_examples.md", k: int = 5) -> None:
    """Save k before/after pairs from the real test set for the report."""
    rows = _short_examples(test_df, k)
    lines = ["# Renaming examples (real test set)", "",
             "Same function before and after `rename_identifiers`. The label does not change.", ""]
    for _, r in rows.iterrows():
        new_code, info = rename_identifiers(r["code"])
        lab = "Vulnerable (1)" if int(r["label"]) == 1 else "Safe (0)"
        lines += [f"## id {int(r['id'])}, label {lab}", "",
                  f"Renamed: {info['n_fun']} function name(s), {info['n_var']} variable name(s). "
                  f"Parse errors before/after: {info['parse_errors_before']}/{info['parse_errors_after']}.", "",
                  "Before:", "", "```c", r["code"], "```", "", "After:", "", "```c", new_code, "```", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path} ({len(rows)} examples)")


if __name__ == "__main__":
    # Run from the repo root after data_prep:  python -m src.renaming
    import json

    from src.data_prep import clean_code, load_splits

    got, _ = _show("copy_name example", EXAMPLE)
    got_clean = clean_code(got)
    print("matches expected:", got_clean == EXPECTED)
    assert got_clean == EXPECTED, f"expected:\n{EXPECTED}\ngot:\n{got_clean}"

    # small edge cases: fields, types, labels, macros, library calls, strings
    edge = ("static void *make(Ctx *c, size_t n) { Buf *b = av_malloc(sizeof(Buf)); "
            "if (!b) goto fail; b->size = n; c->buf = b; memset(b->data, 0, n); "
            "av_log(c, AV_LOG_ERROR, \"n=%d\", (int)n); return b; fail: return NULL; }")
    _show("edge cases (fields, types, goto label, macro, library call)", edge)

    test = load_splits("data/processed", names=("test",))["test"]
    for _, r in _short_examples(test, 2).iterrows():
        _show(f"test id {int(r['id'])} (label {int(r['label'])})", r["code"])

    renamed, stats = build_renamed_split(test)
    assert (renamed["id"].values == test["id"].values).all()
    assert (renamed["label"].values == test["label"].values).all()
    print("renamed test set stats:", json.dumps(stats, indent=2))
    write_examples_md(test, "results/renaming_examples.md", k=5)

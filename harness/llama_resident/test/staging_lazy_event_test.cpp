// White-box staging lifetime regression; no GPU required.
// Compile this translation unit instead of ggml-backend.cpp and link ggml-base.
#include "../../../llama.cpp/ggml/src/ggml-backend.cpp"
#include <cstdio>
#include <cstring>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line=%d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
struct Upload { const uint8_t * src; std::vector<uint8_t> expected; };
struct Mock {
    ggml_backend backend{};
    ggml_backend_device dev{};
    ggml_backend_buffer_type buft{};
    std::vector<Upload> uploads;
    size_t completed = 0;
    int records = 0, waits = 0, freed_events = 0;
};
struct Event { Mock * mock; size_t end = 0; };
static void consume(Mock & m, size_t end) {
    REQUIRE(end <= m.uploads.size());
    while (m.completed < end) {
        const auto & u = m.uploads[m.completed++];
        REQUIRE(std::memcmp(u.src, u.expected.data(), u.expected.size()) == 0);
    }
}
static void record(ggml_backend_t b, ggml_backend_event_t e) {
    auto & m = *(Mock *) b->context;
    ((Event *) e->context)->end = m.uploads.size();
    ++m.records;
}
static void wait_event(ggml_backend_dev_t, ggml_backend_event_t e) {
    auto & ev = *(Event *) e->context;
    ++ev.mock->waits;
    consume(*ev.mock, ev.end);
}
static void sync_backend(ggml_backend_t b) {
    auto & m = *(Mock *) b->context;
    consume(m, m.uploads.size());
}
static ggml_backend_event_t event_new(ggml_backend_dev_t dev) {
    return new ggml_backend_event{dev, new Event{(Mock *) dev->context}};
}
static void event_free(ggml_backend_dev_t dev, ggml_backend_event_t e) {
    ++((Mock *) dev->context)->freed_events;
    delete (Event *) e->context;
    delete e;
}
static void free_buffer(ggml_backend_buffer_t b) {
    auto & m = *(Mock *) b->buft->context;
    auto lo = (uintptr_t) b->context, hi = lo + b->size;
    for (size_t i = m.completed; i < m.uploads.size(); ++i) {
        auto p = (uintptr_t) m.uploads[i].src;
        REQUIRE(p < lo || p >= hi); // no free while the mock DMA still reads this allocation
    }
    std::free(b->context);
}
static void * get_base(ggml_backend_buffer_t b) { return b->context; }
static ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t t, size_t n) {
    ggml_backend_buffer_i iface{};
    iface.free_buffer = free_buffer;
    iface.get_base = get_base;
    return ggml_backend_buffer_init(t, iface, std::malloc(n), n);
}
static void setup(Mock & m) {
    m.buft.context = &m;
    m.buft.iface.alloc_buffer = alloc_buffer;
    m.buft.iface.is_host = [](ggml_backend_buffer_type_t) { return true; };
    m.buft.iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(128); };
    m.dev.context = &m;
    m.dev.iface.get_host_buffer_type = [](ggml_backend_dev_t d) { return &((Mock *) d->context)->buft; };
    m.dev.iface.event_new = event_new;
    m.dev.iface.event_free = event_free;
    m.dev.iface.event_synchronize = wait_event;
    m.backend.context = &m;
    m.backend.device = &m.dev;
    m.backend.iface.event_record = record;
    m.backend.iface.synchronize = sync_backend;
    m.backend.iface.set_tensor_async = [](ggml_backend_t, ggml_tensor *, const void *, size_t, size_t) {};
}
static ggml_backend_sched_t make_sched(Mock & m) {
    auto s = (ggml_backend_sched_t) std::calloc(1, sizeof(ggml_backend_sched));
    s->n_backends = 1;
    s->n_copies = 1;
    s->backends[0] = &m.backend;
    return s;
}
static void enqueue(Mock & m, ggml_backend_sched_t s, uint8_t * p, size_t n, uint8_t value) {
    std::memset(p, value, n);
    m.uploads.push_back({p, std::vector<uint8_t>(n, value)});
    // Same pending transition as compute_splits (including both error exits).
    s->input_staging_pending[0] = true;
}
int main() {
    Mock m; setup(m);
    auto s = make_sched(m);
    ggml_tensor t{}; t.type = GGML_TYPE_I8;
    for (auto & ne : t.ne) ne = 1;
    t.ne[0] = 64; t.nb[0] = 1;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) t.nb[i] = t.nb[i-1]*t.ne[i-1];
    ggml_backend_buffer source{}; source.buft = &m.buft; t.buffer = &source;
    ggml_tensor * inputs[] = {&t};
    ggml_backend_sched_split split{}; split.backend_id = 0; split.n_inputs = 1; split.inputs = inputs;
    auto get = [&]() { auto p = ggml_backend_sched_input_staging(s, &split); REQUIRE(p != nullptr); return p; };
    auto p = get();
    REQUIRE(m.records == 0 && m.waits == 0);
    for (int i = 0; i < 8; ++i) {
        enqueue(m, s, p, 64, uint8_t(i + 1));
        ggml_backend_sched_synchronize(s);
        p = get();
    }
    REQUIRE(m.completed == 8 && m.records == 0 && m.waits == 0);
    enqueue(m, s, p, 64, 31);
    p = get(); // must consume before a later writer can overwrite staging
    REQUIRE(m.completed == 9 && m.records == 1 && m.waits == 1);
    enqueue(m, s, p, 64, 47);
    t.ne[0] = 1024;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) t.nb[i] = t.nb[i-1]*t.ne[i-1];
    p = get(); // buffer growth must first finish the outstanding upload
    REQUIRE(m.completed == 10 && m.records == 2 && m.waits == 2);
    enqueue(m, s, p, 1024, 63);
    ggml_backend_sched_free(s); // pending upload must finish before buffer/event destruction
    REQUIRE(m.completed == 11 && m.records == 3 && m.waits == 3 && m.freed_events == 1);
    std::puts("STAGING_LAZY_EVENT_PASS synchronized=8 reuse=1 growth=1 pending_teardown=1 records=3 exact_uploads=11");
}

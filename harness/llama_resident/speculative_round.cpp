#include "speculative_round.hpp"

#include <stdexcept>
#include <utility>

std::vector<Emit> run_speculative_round(SpeculativeRoundState st, SpeculativeRoundBackend &backend) {
    if (!st.draft || !st.use_ckpt_tgt || !st.use_ckpt_dft || !st.n_past || !st.id_last || !st.prompt_tgt ||
        !st.tokens || !st.last_step_drafts || !st.last_step_hits) {
        throw std::runtime_error("speculative round state incomplete");
    }
    *st.last_step_drafts = 0;
    *st.last_step_hits = 0;

    const int depth = backend.depth();
    auto &draft = *st.draft;
    auto &use_ckpt_tgt = *st.use_ckpt_tgt;

    while (true) {
        if (draft.empty()) {
            backend.prepare_draft(st);
        } else if (use_ckpt_tgt) {
            backend.require_checkpoint_present();
        }

        backend.verify_decode(st);

        auto smpl_save = backend.clone_sampler();

        const size_t n_draft = draft.size();
        const std::vector<int32_t> &ids = backend.greedy_accept(draft);
        if (ids.empty()) {
            throw std::runtime_error("speculative accept produced no tokens");
        }

        *st.last_step_drafts += depth;
        const int64_t attempt_hits = static_cast<int64_t>(ids.size()) - 1;
        *st.last_step_hits += attempt_hits;
        backend.on_verify_attempt(depth, attempt_hits);

        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            backend.rollback_partial_accept(st);
            if (smpl_save) {
                backend.restore_sampler(std::move(smpl_save));
            }
            continue;
        }

        backend.accept_spec_prefix(ids.size() - 1);
        *st.n_past += static_cast<int>(ids.size()) - 1;

        backend.finish_verify_before_emits(st, ids.size());

        std::vector<Emit> out;
        out.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            st.prompt_tgt->push_back(*st.id_last);
            *st.id_last = ids[i];
            st.tokens->push_back(*st.id_last);
            Emit e = backend.emit_logits_at_verify_index(static_cast<int>(i));
            out.push_back(e);
            backend.sampler_accept_token(ids[i]);
        }
        draft.clear();
        backend.trim_kv_tail(*st.n_past);
        backend.on_round_complete();
        return out;
    }
}

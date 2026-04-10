# MGR-SID v2 Proxy Sanity

- Date: 2026-04-11
- Item count: 3686
- Hard-item definition: baseline `l2_leaf_count >= 4`
- Easy-item definition: baseline `l2_leaf_count == 1`

## Main Takeaways

### offline_combined
- usable: `False`
- hard-vs-easy AUC: `0.7186571343665706`
- hard/easy mean gap: `0.0469`
- hard base rate: `0.2580`
- hard rate in top 10% proxy items: `0.6141`
- improved@3 rate in top 10% proxy items: `0.0135`
- worsened@3 rate in top 10% proxy items: `0.0116`
- mean `l2` leaf-count reduction in top 10% proxy items: `0.5679`

### offline_plus_online
- usable: `False`
- hard-vs-easy AUC: `0.5858982106935491`
- hard/easy mean gap: `0.0284`
- hard base rate: `0.2580`
- hard rate in top 10% proxy items: `0.3478`
- improved@3 rate in top 10% proxy items: `0.0023`
- worsened@3 rate in top 10% proxy items: `0.0091`
- mean `l2` leaf-count reduction in top 10% proxy items: `0.9755`

## Proxy Components

- semantic density mean: `0.8953`
- semantic-collab disagreement mean: `0.9804`
- graph competition mean: `0.9633`
- online uncertainty mean: `0.2502`

## Recommendation

- `R001` is **borderline but practically usable** for a first `v2` tokenizer run:
  - offline combined proxy reaches `AUC = 0.7187`
  - hard-item rate in the top `10%` proxy bucket rises from `0.2580` to `0.6141`
  - `improved@3` is slightly higher than `worsened@3` in the top proxy bucket
- `R002` is **not recommended** in its current form:
  - adding the current online uncertainty weakens hard/easy separation
  - the top proxy bucket becomes more enriched for `worsened@3` than `improved@3`
- Practical next step:
  - launch the first `v2` tokenizer run with the **offline combined ambiguity prior only**
  - keep the current online uncertainty term out of the first `v2` main experiment

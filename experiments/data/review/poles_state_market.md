# Pole pairs: state_market

## YAML block (ready to paste into value_axes.yaml)

```yaml
state_market:
  en_pairs:
    - [planning, competition]
    - [subsidy, profit]
    - [intervention, laissez-faire]
    - [protectionism, liberalization]
    - [nationalization, privatization]
    - [dirigisme, spontaneity]
    - [taxation, exchange]
    - [welfare, enterprise]
    - [tariff, trade]
    - [command, bargain]
  zh_pairs:
    - [计划, 市场]
    - [调控, 竞争]
    - [补贴, 利润]
    - [国有, 民营]
    - [干预, 放任]
    - [统制, 自由]
    - [公有, 私营]
    - [配给, 交易]
    - [产业政策, 市场机制]
    - [统购, 议价]
```

## Rationale (one short paragraph per pair)

### EN pairs

1. `[planning, competition]` — The canonical Hayek–Mises opposition (Hayek, *The Road to Serfdom*, 1944; *The Use of Knowledge in Society*, 1945). Central planning as epistemic substitute for the price signal versus decentralized competitive discovery. The pair isolates the economic-ordering dimension: both "planning" and "competition" are regulatory configurations, neither reduces to a legal-classification contrast.

2. `[subsidy, profit]` — Subsidy is the paradigmatic instrument of fiscal redirection (EU state-aid jurisprudence, TFEU art. 107; WTO SCM Agreement); profit is the market's self-allocating residual. The pair captures the state-as-allocator versus market-as-allocator distinction with minimal overlap with rights/duties or public/private classification.

3. `[intervention, laissez-faire]` — Textbook Polanyian opposition (*The Great Transformation*, 1944) and the exact axis along which Cowperthwaite's "positive non-interventionism" was articulated for Hong Kong. Both poles are second-order stances on economic governance, not legal categories.

4. `[protectionism, liberalization]` — Trade-policy register (GATT/WTO, Bhagwati, *In Defense of Globalization*). Protectionism marks state shielding of domestic actors; liberalization marks withdrawal of state barriers to exchange. Distinct from public/private because both actors remain legally private firms; what changes is the regulatory perimeter.

5. `[nationalization, privatization]` — The property-form axis of political economy (Sappington & Stiglitz 1987 on privatization; Kornai on socialist transition). Directly bears on PRC state-owned-enterprise doctrine versus HK/Anglo privatization waves. Sharp, widely lexicalized in corporate and constitutional economics.

6. `[dirigisme, spontaneity]` — French-tradition *dirigisme* (post-war planning under Monnet) versus Hayekian "spontaneous order" (*Law, Legislation and Liberty*, vol. I). High-register but clean antonymy in the specific sense of directed economic order versus emergent order. Short single tokens, sentence-encoder-friendly.

7. `[taxation, exchange]` — Taxation is the state's non-consensual extraction; voluntary market exchange is its consensual counterpart (Buchanan, *The Power to Tax*, 1980). The pair anchors the axis in fiscal sociology without drifting into rights/duties (which would pair "tax" with "obligation").

8. `[welfare, enterprise]` — Welfare-state transfers (Esping-Andersen, *Three Worlds of Welfare Capitalism*) versus private enterprise as productive unit. Captures the redistributive-versus-productive configuration. "Enterprise" appears in `public_private` only as "administration / enterprise", so register is distinct (welfare is not administrative apparatus).

9. `[tariff, trade]` — Classical political-economy pair (Ricardo, Smith, modern trade jurisprudence). Tariff as emblem of state border-regulation; trade as emblem of market flow. Clean tokens, unambiguous economic register.

10. `[command, bargain]` — The Williamsonian and Coasean opposition between hierarchical coordination and price-based exchange (Williamson, *Markets and Hierarchies*, 1975). Also invokes "command economy" shorthand. Both are short nouns with crisp embedding signatures.

### ZH pairs

1. `[计划, 市场]` — 计划经济 vs 市场经济. Foundational opposition in PRC constitutional-economic discourse since the 1993 Constitutional Amendment transitioning to 社会主义市场经济. The most canonical realization of the axis in Chinese legal-political vocabulary.

2. `[调控, 竞争]` — 宏观调控 (macro-regulation, an explicit article-level PRC constitutional concept) versus 竞争 (competition, anchor of the 反垄断法 Anti-Monopoly Law 2008). Captures state steering of macro variables against market competitive dynamics.

3. `[补贴, 利润]` — Subsidy versus profit. 补贴 is the operative term in PRC industrial-policy discourse and in WTO disputes over Chinese state aid; 利润 is the market's residual. Direct economic-allocator contrast.

4. `[国有, 民营]` — 国有企业 versus 民营企业: the property-form distinction at the heart of PRC corporate structure and mixed-ownership reform (混合所有制). Thickly lexicalized; detects exactly the cross-tradition variation the axis is designed to capture.

5. `[干预, 放任]` — State intervention versus laissez-faire. 放任 is the Chinese lexical equivalent of laissez-faire and appears in classical liberal translations (Yan Fu's rendering of Adam Smith). Crisp antonymy along the axis.

6. `[统制, 自由]` — 统制经济 (controlled/statist economy, Republican-era term still current in economic history) versus 自由 economy. "Self-reliance"-style pairings are avoided (already in `individual_collective`); here 自由 is scoped by 统制 into the economic register.

7. `[公有, 私营]` — Public ownership versus private operation. Note 公有 is distinct from 公共 (already in `individual_collective` and `public_private`): 公有 specifically denotes property ownership form, not public-domain classification. Paired with 私营 (private-operated firm) rather than 私人 for the same reason.

8. `[配给, 交易]` — Rationing/allocation versus transactional exchange. 配给 captures the command-economy distributive mode (wartime and Mao-era 配给制); 交易 is the generic market transaction. Both are concrete nouns, unambiguous.

9. `[产业政策, 市场机制]` — Industrial policy versus market mechanism. Both are settled four-character economic-policy terms in PRC discourse; their opposition structures much of contemporary Chinese regulatory debate (e.g. Lin Yifu vs Zhang Weiying on industrial policy). The longest token pair but lexically stable.

10. `[统购, 议价]` — State monopsony purchasing (统购统销, the Maoist grain-procurement system) versus negotiated/bargained price. Historically marked but still a recognizable pair in economic-legal Chinese, and it matches the EN `[command, bargain]` conceptually without being a translation.

## Known trade-offs and residual ambiguities

Pair EN-4 (`[protectionism, liberalization]`) and pair ZH-5 (`[干预, 放任]`) both lean on the contested concept of "liberalization" / "放任". Neither is semantically clean: liberalization in WTO jurisprudence is a process-noun whose embedding may drift toward generic political reform rather than economic ordering. I retained it because the doctrinal signal (removal of state-erected barriers) is exactly what the axis needs and alternatives (e.g. "openness") are even more diffuse. The residual noise should be absorbed by the averaging across ten pairs per Kozlowski.

Pair EN-10 (`[command, bargain]`) risks a rights/duties echo because "command" also figures in imperative-theory jurisprudence (Austin). However, the paired opposite "bargain" shifts the dominant embedding context toward Williamsonian hierarchy-versus-exchange rather than sovereign-command-versus-subject-duty. I kept it because the political-economy literature on "command economy" gives the pair a clear economic-regulatory anchor.

Pair ZH-7 (`[公有, 私营]`) deliberately uses 公有 rather than 公共 to avoid collision with `individual_collective` and `public_private`. 公有 denotes ownership form (owned by the public/state) while 公共 denotes public domain; the distinction is fine but stable in PRC legal-economic discourse. Sentence-encoder tokenization should preserve it.

Pair ZH-10 (`[统购, 议价]`) is the most historically marked pair. In contemporary PRC vocabulary 统购 primarily survives as an economic-history term, but its embedding signature remains clearly in the state-procurement cluster and its antonym 议价 (negotiated price) is fully current. I accepted the historical tint because it sharpens the axis at the state-direction pole.

## Vocabulary-overlap audit

Single-word overlaps with the three existing axes:

- EN `laissez-faire` — no token overlap with any existing pair.
- EN `enterprise` — appears in `public_private` as `[administration, enterprise]`. Here it is paired with `welfare`, not `administration`, shifting the contrast from legal-form (administrative vs commercial entity) to redistributive-configuration (welfare-state transfers vs productive private enterprise). Justified.
- EN `competition` — no overlap.
- EN `market` — not used as a pole here (avoided; `public_private` has `[bureaucracy, market]`). The ZH `市场` appears in `public_private` as `[官僚, 市场]` but here `市场` pairs with `计划`, the canonical economic-ordering opposition rather than the bureaucracy/market classification contrast. Justified because removing 市场 from the state_market axis would be doctrinally absurd (it is the axis name).
- EN `trade` / `tariff` — no overlap.
- EN `command` — no overlap.
- EN `subsidy`, `profit`, `taxation`, `exchange`, `nationalization`, `privatization`, `protectionism`, `liberalization`, `dirigisme`, `spontaneity`, `bargain`, `welfare`, `planning`, `intervention` — no overlap.
- ZH `自由` — appears in `individual_collective` as `[自由, 服从]` and in `rights_duties` as `[自由, 责任]`. Here paired with `统制`, scoping it firmly into the economic-ordering register (controlled-economy vs free-economy). Overlap is tolerable because the averaging across pairs within each axis will dilute any cross-axis bleed, and Kozlowski's method is robust to single-word reuse when the paired opposite is different (cf. Kozlowski et al. 2019, §3.2 on multi-pair robustness).
- ZH `市场` — see above; indispensable to the axis concept.
- ZH `个人`, `公共`, `私人` — deliberately NOT reused. Used 民营, 私营, 公有 instead to preserve axis identity.
- ZH `干预` — no overlap.
- ZH `调控`, `竞争`, `补贴`, `利润`, `国有`, `放任`, `统制`, `配给`, `交易`, `产业政策`, `市场机制`, `统购`, `议价` — no overlap.

No overlap requires rejection. The three single-word reuses (`enterprise`, `市场`, `自由`) are doctrinally necessary and their paired opposites pull them cleanly into the state_market register.

# ACE Architecture

This document explains how the **Agentic Context Engineering (ACE)** framework
is structured and how a run flows through it. It is the engineering companion to
the paper *"Agentic Context Engineering: Evolving Contexts for Self-Improving
Language Models"* (ICLR 2026).

> **TL;DR** — ACE treats an LLM's context as an evolving **playbook** of small,
> itemized **bullets**. A **Generator** solves a query, a **Reflector** distills
> reusable lessons, and a **Curator** emits **incremental delta operations** that
> are merged by deterministic (non-LLM) logic. A **grow-and-refine** step keeps
> the playbook compact. This avoids *brevity bias* and *context collapse*.

---

## 1. The big picture

```mermaid
flowchart LR
    Q([Query]) --> G[Generator]
    PB[(Context Playbook)] -. injected .-> G
    G -->|trajectory + bullet usage| R[Reflector]
    FB([Feedback:<br/>labels or execution signal]) --> R
    R -->|insights<br/>iterative refinement| C[Curator]
    C -->|delta context items| M{{Deterministic Merge<br/>non-LLM}}
    M --> PB
    M --> GR[Grow & Refine<br/>dedupe / prune]
    GR --> PB

    classDef role fill:#1e293b,color:#fff,stroke:#0f172a;
    classDef store fill:#2563eb,color:#fff,stroke:#1e40af;
    classDef det fill:#16a34a,color:#fff,stroke:#15803d;
    class G,R,C role;
    class PB store;
    class M,GR det;
```

The three **roles** are LLM-backed and specialized; the **merge** and
**grow-and-refine** steps are plain, auditable Python. That separation is the
heart of the design: the model only ever *proposes localized edits*, so
accumulated knowledge can never be silently erased by a runaway rewrite.

---

## 2. The two failure modes ACE fixes

```mermaid
flowchart TB
    subgraph Prior["Prior context adaptation"]
        direction TB
        BB[Brevity bias<br/>optimizers collapse to short, generic prompts]
        CC[Context collapse<br/>full LLM rewrite compresses & erases detail]
    end
    subgraph ACEfix["ACE"]
        direction TB
        IDU[Incremental delta updates<br/>localized add / update / remove]
        GRF[Grow-and-refine<br/>append + in-place update + semantic dedupe]
        REF[Dedicated Reflector<br/>separates evaluation from curation]
    end
    BB -. solved by .-> REF
    BB -. solved by .-> IDU
    CC -. solved by .-> IDU
    CC -. solved by .-> GRF
```

`examples/02_context_collapse.py` reproduces context collapse with a
`MonolithicRewriteAgent` and shows ACE staying collapse-free.

---

## 3. The adaptation step (sequence)

```mermaid
sequenceDiagram
    participant U as Caller
    participant E as ACE.step()
    participant G as Generator (LLM)
    participant R as Reflector (LLM)
    participant C as Curator (LLM)
    participant D as apply_delta() (det.)
    participant P as Playbook

    U->>E: step(sample, feedback)
    E->>P: render() context
    E->>G: generate(sample, playbook)
    G-->>E: answer + helpful/harmful bullet ids
    E->>R: reflect(sample, gen, feedback)
    Note over R: up to N refinement rounds
    R-->>E: insights + diagnosis
    E->>C: curate(insights)
    C-->>E: DeltaContext (ADD/UPDATE/REMOVE)
    E->>D: apply_delta(playbook, delta)
    D->>P: localized edits + counter updates
    E->>P: grow_and_refine() (proactive or lazy)
    E-->>U: StepRecord (metrics, deltas, timing)
```

---

## 4. Offline vs. online adaptation

```mermaid
flowchart LR
    subgraph Offline["Offline (e.g. system-prompt optimization)"]
        direction TB
        TR[(Train split)] --> EP{Multi-epoch loop}
        EP -->|revisit samples| STEP1[ACE.step]
        STEP1 --> EP
        EP --> PBO[(Playbook)]
    end
    subgraph Online["Online (test-time memory adaptation)"]
        direction TB
        S[Next sample] --> PRED[predict with current playbook]
        PRED --> LEARN[ACE.step updates playbook]
        LEARN --> S
    end
    PBO -. optional warm start .-> Online
```

- **Offline** (`ACE.adapt_offline`): multiple epochs over a training split to
  progressively strengthen the playbook. Optionally uses ground-truth labels.
- **Online** (`ACE.adapt_online`): for each test sample, predict first, then
  learn from the *same* trajectory and feedback. Can be warm-started from an
  offline playbook (the paper's strongest AppWorld configuration).

---

## 5. Data model

```mermaid
classDiagram
    class Bullet {
        +str id  "ctx-xxxxxxxx"
        +str content
        +str section
        +int helpful_count
        +int harmful_count
        +list~str~ tags
        +int created_at_step
        +score() int
        +render() str
    }
    class Playbook {
        +list~str~ sections
        +add(Bullet)
        +update(id, content)
        +remove(id)
        +mark_helpful(id)
        +mark_harmful(id)
        +render() str
        +stats() dict
    }
    class DeltaOperation {
        +DeltaOp op  "ADD|UPDATE|REMOVE"
        +str section
        +str content
        +str target_id
    }
    class DeltaContext {
        +list~DeltaOperation~ operations
        +list~str~ helpful_ids
        +list~str~ harmful_ids
    }
    Playbook "1" o-- "many" Bullet
    DeltaContext "1" o-- "many" DeltaOperation
    DeltaContext ..> Playbook : apply_delta()
```

A **bullet** is the atomic unit (akin to a memory entry in Dynamic Cheatsheet /
A-MEM, plus counters). Bullets are grouped into **sections**
(`strategies`, `domain_concepts`, `common_mistakes`, `tool_usage`,
`formatting` by default). The Generator references bullet **ids** so updates are
*localized*.

---

## 6. Module map

| Module | Responsibility |
| --- | --- |
| `ace/playbook.py` | `Bullet`, `Playbook` — the evolving, sectioned context |
| `ace/delta.py` | `DeltaOperation`, `DeltaContext`, `apply_delta` — deterministic merge |
| `ace/roles.py` | `Generator`, `Reflector`, `Curator` + their prompts |
| `ace/refine.py` | `grow_and_refine` — semantic dedupe + harmful-bullet pruning |
| `ace/engine.py` | `ACE` orchestrator, `adapt_offline` / `adapt_online`, `StepRecord` |
| `ace/llm.py` | `LLM` protocol, `OpenAILLM`, deterministic `SimulatedLLM` |
| `ace/feedback.py` | `Feedback` — labeled or label-free execution signals |
| `ace/tasks.py` | `Sample`, `Task`, `TeachingEnvironment` (offline benchmark) |
| `ace/baselines.py` | `StaticAgent`, `MonolithicRewriteAgent` (context collapse) |
| `ace/visualize.py` | `LiveRunVisualizer` (terminal), `render_html_report` (HTML) |
| `ace/integrations/openai_agents.py` | `ACEAgent` — OpenAI Agents SDK memory |
| `ace/cli.py` | `ace` command-line entrypoint |

---

## 7. Why incremental deltas are cheap

```mermaid
flowchart LR
    subgraph Mono["Monolithic rewrite"]
        direction TB
        M1[Read whole context] --> M2[LLM rewrites everything]
        M2 --> M3[Risk: compress & collapse]
        M2 --> M4[Cost ∝ full context, every step]
    end
    subgraph Delta["ACE delta update"]
        direction TB
        D1[Reflect on one trajectory] --> D2[Emit a few bullet edits]
        D2 --> D3[Deterministic merge - no LLM]
        D2 --> D4[Cost ∝ delta size]
    end
```

Because the merge is non-LLM and operations are itemized:

- multiple deltas can be merged **in parallel** (batched adaptation);
- adaptation cost scales with the **delta**, not the whole context;
- long contexts amortize well at serve time via **KV-cache reuse**.

The paper reports up to **−86.9%** adaptation latency, **−75.1%** rollouts
(offline AppWorld vs GEPA), and **−83.6%** token cost (online FiNER vs Dynamic
Cheatsheet). `examples/03_offline_vs_online.py` illustrates the delta-vs-rewrite
token-ingestion gap on the bundled teaching environment.

---

## 8. Extending ACE

- **New backend** — implement the two-method `LLM` protocol (`complete`,
  `complete_json`) and pass it to `ACE(...)`.
- **New task** — build a `Task` with your own samples and an `evaluate` scorer,
  or wrap a live environment and feed `Feedback(signal=...)` for the label-free
  path.
- **New agent framework** — mirror `ace/integrations/openai_agents.py`: inject
  `playbook.render()` into the system prompt and call `ace.step(...)` with the
  captured trajectory.
- **Semantic dedupe** — pass `embedder=make_openai_embedder()` (or any batched
  embedding callable) to `ACE(...)` for embedding-based de-duplication.

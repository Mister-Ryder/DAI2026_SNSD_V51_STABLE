# Paper-ready method draft: Semantic-Safeguarded Tri-Graph Joint Negotiation (v5.1)

This document is written as reusable English material for the method section of
the DAI submission. It describes the frozen v5.1 implementation. The term
*exact* below refers to full-graph feasibility verification and exact objective
comparison over a finite candidate union; it does **not** mean that the full
combinatorial problem is solved to global optimality.

We refer to the unified algorithm as **SNSD-v5.1**. The frozen primary
instantiation is labelled **SNSD-v5.1 phased
(`AlternatingConflictBenefit`)**, because it temporally separates conflict and
benefit channels. `Full` is the simultaneous-channel instantiation of the same
executor, and `ShuffledComm` is a correspondence control. This naming does not
imply that the phased schedule universally dominates either control.

## 1. Problem formulation

We consider satellite--ground link scheduling over a fixed planning horizon.
Let each feasible communication opportunity be a vertex $v\in V$, with
nonnegative reward $w_v$ equal to its link duration. Two vertices are joined
by an undirected edge when their links cannot be scheduled simultaneously due
to a ground-station or satellite-resource conflict. The resulting problem is a
maximum-weight independent set (MWIS):

\[
\max_{x\in\{0,1\}^{|V|}} F(x)=\sum_{v\in V}w_vx_v,
\qquad
x_u+x_v\leq 1,\quad \forall (u,v)\in E.
\]

The vertices are partitioned by link start time into $K$ disjoint groups,
$V=\dot\bigcup_{i=1}^{K}V_i$. Agent $i$ owns the induced local subproblem
on $V_i$. We write

\[
E_i=\{(u,v)\in E:u,v\in V_i\},\qquad
E_{\mathrm{cut}}=E\setminus\bigcup_i E_i.
\]

The formal C3 implementation uses six time agents. Its canonical graph has
69,923 vertices, 979,960 conflict edges, and 8,980 cut edges. The decomposition
is a chain on C3, but the coalition repair is implemented on connected
components of the active agent-interaction graph and is not restricted to a
single pair of agents.

Each agent calls the same fixed local worker under the same query budget. The
worker returns a realized local independent set rather than an assumed local
optimum. The multi-agent method is therefore responsible for deciding which
local states to explore, communicating boundary structure, composing realized
records, and repairing cross-agent incompatibilities. All methods in the
comparison use the same local worker and paid-query schedule.

## 2. Three coupled graph views

The central design principle is to avoid a second information bottleneck after
decomposition. Partitioning already removes most of the full combinatorial
context from each agent. Compressing every local state once more into a single
scalar or pooled embedding makes it difficult to recover cross-partition
substitutions and refills. We instead maintain three coupled views.

### 2.1 Private local subgraphs

Agent $i$ searches the induced graph $G_i=(V_i,E_i)$ and retains a small
set of *realized elites*, not only its current anchor. A realized elite contains
the complete selected-vertex set returned by the local worker, its exact local
reward, and its boundary signature. These records preserve the high-dimensional
within-group structure needed to construct a response after a boundary vertex
is deleted.

### 2.2 Dynamic boundary interaction graph

The boundary view is induced by the real cut conflicts and the candidate
vertices exposed by the local elites. It carries vertex identities rather than
only pooled statistics. A boundary packet reports explicit cut-conflict
endpoints, selected boundary IDs, candidate roles, and response-relevant local
quantities. In the current implementation, cut-conflict edges are transmitted
explicitly; local substitution/complement structure is represented by candidate
roles and typed relation counts, while exact compatibility is reconstructed
from the canonical conflict graph by the executor.

The boundary view is dynamic because its vertices, active cut edges, prices,
and candidate responses depend on the current private elites. Small connected
components of the active agent-interaction graph define the coalitions that
jointly repair their incompatible private states.

### 2.3 Public global graph

The complete conflict graph supplies information that no local or boundary
view can safely approximate: long-range feasibility, the true scheduling
objective, and the factor residuals used to build pressure, conflict, and
opportunity channels. It also implements the final safety shield. The global
graph guides and verifies coordination, but it does not replace private
subgraph search or reveal future worker outcomes.

The roles of the views are consequently asymmetric:

\[
\text{local subgraphs generate responses}
\ \longrightarrow\
\text{boundary graph negotiates exchanges}
\ \longrightarrow\
\text{global graph verifies and commits}.
\]

## 3. Innovation I: structured tri-graph primal--dual messaging

### 3.1 Structure-preserving boundary packets

At a late negotiation state, agent $i$ retains up to $T$ highest-valued
unique realized records and constructs a packet

\[
P_i=(h_i,\mathcal E_i,B_i,C_i,\Phi_i,E_i^{\mathrm{cut}},R_i).
\]

Here, $h_i$ is the hash of the private anchor, $\mathcal E_i$ lists the
retained elite IDs, $B_i$ is the set of selected boundary vertices, $C_i$
is a bounded set of real candidate IDs, $\Phi_i$ contains candidate features,
$E_i^{\mathrm{cut}}$ contains incident cut conflicts, and $R_i$ contains
typed relation statistics. For each candidate $v\in C_i$, the packet retains

\[
\phi_i(v)=
(f_v,w_v,\widehat\Delta_v,d_v,\widehat r_v,u_v,\rho_v),
\]

where $f_v$ is its elite-selection frequency, $w_v$ is its raw reward,
$\widehat\Delta_v$ is an empirical local marginal, $d_v$ is an estimated
deletion cost, $\widehat r_v$ is the best estimated refill gain, $u_v$ is
the empirical uncertainty, and $\rho_v\in\{\texttt{keep},
\texttt{add-or-refill}\}$ is its role. Packet IDs are traceable to realized
records or deterministic same-partition neighbor expansion. Subsequent refill
actions draw from real one-/two-hop neighborhoods and retained elite records;
no candidate or label is manufactured from a future macro-round outcome.

This representation is deliberately sparse but not scalar. It preserves the
identity of a boundary action and enough local context to construct a complete
subgraph response when that action is considered by another agent.

### 3.2 Semantic channels

For a packet candidate $v\in V_i$, the inherited message pipeline exposes
three channels: pressure $p_i(v)$, conflict $c_i(v)$, and opportunity
$o_i(v)$. Values are normalized independently within each agent by the
maximum absolute value. The full semantic keep credit is

\[
q_i^{\mathrm{full}}(v)=
\operatorname{norm}_i\!\left[p_i(v)+c_i(v)-\tfrac12|o_i(v)|\right].
\]

The zero-semantic control sets $q_i(v)=0$. The registered channel controls
are

\[
q_i^{\mathrm{conf}}(v)=\operatorname{norm}_i[p_i(v)+c_i(v)],
\qquad
q_i^{\mathrm{benefit}}(v)=\operatorname{norm}_i[p_i(v)-o_i(v)].
\]

For the shuffled control, the next agent's ordered channel values are reassigned
position-wise to the target agent's sorted packet IDs. This preserves the
number of transmitted values and the evaluation budget while destroying their
agent/vertex correspondence.

### 3.3 Boundary prices and joint deletion decisions

Let $S^{\mathrm{priv}}$ be the union of the current private elites and let
$E_{\mathrm{cut}}(S^{\mathrm{priv}})$ denote its active cut conflicts. During
microcycle $m$, every active edge $e=(u,v)$ receives the nonnegative price

\[
\lambda_e^{m+1}=
\max\left\{0,\lambda_e^m+
\frac{0.20}{\sqrt{m+1}}
\max\left(1,\frac{w_u+w_v}{2}\right)\right\}.
\]

The incident price of vertex $v$ is
$\Lambda_v=\sum_{e\ni v}\lambda_e$. For an active conflict, the executor
compares the two endpoint keep scores

\[
k_i(v)=w_v-\min\{w_v,\widehat r_v\}
+0.35\,s_{uv}q_i(v)-\Lambda_v,
\qquad
s_{uv}=\max\left(1,\frac{w_u+w_v}{2}\right).
\]

The lower-scoring endpoint is tentatively deleted, with deterministic
reward/ID tie breaking. Repair is organized in bounded deletion batches inside
connected two-/three-agent coalition components; if conflicts remain after one
batch, another deterministic batch is opened. Thus all active cut conflicts
are resolved before a response is admitted. The prices are a discrete
non-convex coordination heuristic. We make no ADMM or asymptotic convergence
claim.

### 3.4 Coalitional delete--replace--refill closure

Deleting the cheaper endpoint is insufficient because its reward may only be
recoverable through a coordinated local response. For every affected agent,
the executor forms a refill pool from (i) same-partition one- and two-hop
neighbors of deleted vertices and (ii) vertices appearing in retained elites.
It greedily inserts globally compatible candidates according to

\[
r_i(v)=\frac{w_v}{1+\log(1+\deg(v))}+0.20w_vq_i(v).
\]

It then attempts deterministic one-for-many exchanges: a new vertex replaces
its conflicting selected vertices only if all of those conflicts are local and
its reward exceeds their total reward. Thus *replace* is not an abstract label;
it is an executable local exchange following a coalition-level boundary
deletion. The output is a complete selected set for every agent, not merely a
new search anchor. The full graph finally verifies independence and evaluates
the true reward.

## 4. Innovation II: paired semantic safeguarding and counterfactual coalition responses

### 4.1 Same-state $C_0/C_A$ branches

An unprotected semantic response can be rejected at the current decision yet
still enter the next candidate domain, where it changes later search paths.
This creates semantic *domain pollution*. v5.1 removes this path by evaluating
two branches from the same immutable tuple of private elites, current records,
joint records, graph state, and message channels:

- $C_0$: a coalition response generated with zero semantic credit;
- $C_A$: a coalition response generated with the registered semantic
  treatment (full, alternating, one-shot, or shuffled).

When the treatment is already zero, $C_A\equiv C_0$ and the second execution
is deduplicated. Otherwise the two branches differ only through the semantic
credit supplied to the structured executor at that state.

### 4.2 Counterfactual microresponses

Within each branch, microcycle zero reproduces the original structured
response. Later microcycles consume the preceding response and deliberately
construct alternatives. If a boundary endpoint was deleted before, it receives
a keep bonus $+5n_v^{\mathrm{del}}$; if a vertex was previously used as a
refill, it receives a refill-ranking penalty $-3n_v^{\mathrm{refill}}$. Hence

\[
\widetilde q_i^m(v)=q_i(v)
+5n_v^{\mathrm{del}}-3n_v^{\mathrm{refill}}.
\]

This feedback is a deterministic diversification control rather than an
approximation of the scheduling objective. Every distinct response is checked
for global feasibility, and the highest true-reward response in the branch is
retained. Consequently repeated microcycles form a portfolio of counterfactual
delete/refill plans instead of repeatedly repricing one unchanged action.

### 4.3 Exact candidate union and semantic gate

At every late decision, the coordinator has the following globally verified
candidates:

- $Y_t$: the current reference schedule;
- $J_t$: the exact factor/chain composition of the currently realized local
  records;
- $R_t^0$: a common message-independent repair of the private-elite union;
- $R_t^m$: the inherited message-conditioned repair, when peer communication
  is available;
- $C_t^0$: the best zero-semantic coalition response;
- $C_t^A$: the best treated-semantic coalition response.

The treated branch is eligible only when it strictly improves on every
nonsemantic alternative:

\[
\operatorname{eligible}(C_t^A)
\iff
F(C_t^A)>
\max\{F(J_t),F(R_t^0),F(R_t^m),F(C_t^0)\}.
\]

For factor $B=1$, the finite exact union is

\[
\mathcal U_t=
\{Y_t,J_t,R_t^0,R_t^m,C_t^0\}
\cup
\{C_t^A:\operatorname{eligible}(C_t^A)\}.
\]

For $B=0$, $C_t^0$ and $C_t^A$ are still computed for paired diagnosis
but neither is admitted to the candidate union. Missing peer-conditioned
repairs are simply omitted. The committed schedule is

\[
Y_{t+1}=\arg\max_{Y\in\mathcal U_t:\,Y\text{ verified feasible}}F(Y),
\]

with deterministic ordering and strict-improvement tie handling. Because
$Y_t\in\mathcal U_t$,

\[
F(Y_{t+1})\geq F(Y_t).
\]

This is a monotonicity guarantee for the committed sequence, not an optimality
guarantee for MWIS.

The future coordination domain follows a separate safeguard. $R_t^0$ and an
available $R_t^m$ remain common repair records. Under $B=1$, $C_t^0$ is
always available as the protected coalition baseline, whereas $C_t^A$ enters
the next domain only if it both passes the strict gate and is selected as the
current exact anchor. Components of a rejected $C_t^A$ are never injected.

## 5. Conflict--benefit alternating protocol

The `AlternatingConflictBenefit` configuration temporally separates the two
message functions while preserving the local-worker budget and structured
executor. For the late $C_A$ branch, odd source steps use
$q^{\mathrm{conf}}$ and even source steps use $q^{\mathrm{benefit}}$. The
first late source step is step 5 in the zero-based implementation, so the late
sequence begins with conflict-only semantics and then alternates with
benefit-only semantics. The inherited proposal pathway follows the same
odd-conflict/even-benefit schedule after its public initial step.

This control tests whether conflict resolution and reward recovery should be
consumed simultaneously or in separate negotiation phases. `Full` uses the
combined semantic credit at every eligible late state. `OneShotComm` exposes
real structured semantics only at the first late state and uses $C_0$
thereafter. `ShuffledComm` preserves payload size and computation but breaks
message ownership. These are protocol definitions, not claims that any one
schedule is universally superior.

There are two related but different estimands. The within-state quantity
$F(C_A)-F(C_0)$ isolates the executor's immediate semantic intervention. The
end-to-end factorial effect of $A$ also includes earlier message-induced
changes to worker proposals and the realized candidate trajectory. These two
effects must not be conflated in the empirical analysis.

## 6. Full algorithm

**Algorithm 1: Semantic-safeguarded tri-graph joint negotiation**

```text
Input: conflict graph G=(V,E,w), partitions {V_i}_{i=1}^K,
       R macro rounds, D proposal steps, packet width T,
       M primal-dual microcycles, fixed local workers, message source
Output: a globally verified schedule Y

1  Y <- globally verified initial schedule
2  for macro round r = 1,...,R do
3      initialise each agent's private elite and realized-record domain from Y
4      for proposal step t = 1,...,D do
5          query every local worker once in parallel under the registered anchor
6          update the private elites and realized local record tables
7          run the factor/chain coordinator to obtain J and boundary residuals
8          if t belongs to the first six private-exploitation steps then
9              update the next local anchors without structured coalition repair
10             continue
11         end if
12         construct the common repair R0
13         construct Rm from the registered peer message when available
14         freeze state S=(private elites, current records, J, graph, channels)
15         C0 <- BRANCH(S, zero semantics, T, M)
16         CA <- BRANCH(S, registered semantics, T, M)
17               [deduplicate line 16 when registered semantics are zero]
18         if B=1 then
19             U <- {Y, J, R0, Rm, C0}
20             if F(CA) > max{F(J),F(R0),F(Rm),F(C0)} then
21                 U <- U union {CA}
22             end if
23         else
24             U <- {Y, J, R0, Rm}
25         end if
26         verify every member of U on G and set Y <- highest-reward member
27         inject C0 only when B=1; inject CA only when CA is the accepted Y
28     end for
29 end for
30 return Y
```

**Algorithm 2: Counterfactual structured branch**

```text
BRANCH(S, semantic mode a, packet width T, microcycles M)
1  build Top-T structure-preserving packets {P_i} from the frozen state S
2  identify active cut conflicts in the union of private elites
3  initialise edge prices lambda <- 0 and response portfolio Q <- empty
4  initialise deleted-endpoint and used-refill counters to zero
5  for microcycle m = 0,...,M-1 do
6      update active cut-edge prices with step size 0.20/sqrt(m+1)
7      form semantic credit q^m from mode a and prior response counters
8      delete conflict endpoints using reward, refill, credit, and dual price
9      greedily refill affected subgraphs and attempt local one-for-many exchange
10     if the complete response is globally feasible and structurally new then
11         append it to Q with its true objective
12     end if
13     update deleted-endpoint and used-refill counters from this response
14 end for
15 return the highest-true-objective response in Q
```

In the registered formal protocol, $K=6$, $D=12$, and $R=4$, giving
$6\times12\times4=288$ paid local-worker calls per run. Lines 12--27 and both
counterfactual branches are coordination-only and add zero paid worker calls.
There is no early stop after a non-improving negotiation step.

## 7. Computation and communication cost

Let $T_i\leq T$ be the number of retained elites for agent $i$,
$c_i\leq4T$ the number of packet candidates, $b_i$ the number of selected
boundary IDs, and

\[
r_i=r_i^{\mathrm{conf}}+r_i^{\mathrm{sub}}+
r_i^{\mathrm{comp}}+r_i^{\mathrm{coal}}
\]

the recorded typed-relation count. The implementation charges one packet as

\[
L_i=8c_i+3r_i+T_i+b_i+6
\]

numeric slots. IDs and hashes are conservatively charged as eight-byte slots.
For $M$ microcycles, one branch reports

\[
L_{\mathrm{branch}}
=M\sum_i L_i
+2\sum_{m=0}^{M-2}
\left(|D_m\cap B|+|H_m|\right),
\]

where $D_m\cap B$ is the set of deleted active-boundary endpoints and $H_m$
is the refill set whose counters are fed to the next microcycle. The reported
byte estimate is $8L_{\mathrm{branch}}$. A treated v5.1 decision evaluates
both $C_0$ and $C_A$; an $A=0$ decision deduplicates the identical second
branch. This is a deterministic payload-accounting convention, not a claim
about the byte-level overhead of a particular network serializer.

Packet extraction costs

\[
O\!\left(\sum_i T_i|B_i|+\sum_i c_i^2+H_{\mathrm{packet}}\right),
\]

where $H_{\mathrm{packet}}$ is the adjacency work used to construct local
one-/two-hop refill estimates. A microcycle costs

\[
O\!\left(|E_{\mathrm{cut}}(S^{\mathrm{priv}})|
+H_{\mathrm{refill}}+|V|+|E|\right),
\]

where $H_{\mathrm{refill}}$ is the actual local-neighborhood scan and the last
term is the authoritative full-graph verification. The implementation caps a
refill candidate pool at 192 vertices after neighborhood construction and the
one-for-many exchange shortlist at 64 vertices. At the final union, at most six
named candidates are verified. Since $T\in\{4,6\}$ and
$M\in\{3,4\}$ in the frozen implementation, these coordination costs are
bounded independently of the paid local-worker query count, although their
wall-clock time is reported separately.

## 8. Learned and analytic modes

The v5.1 executor itself is not trained end to end. In **Learned** runs, it
loads the existing validation-selected TriGraph checkpoint from the preceding
protocol and uses that fixed model in the inherited proposal/message pipeline;
the resulting realized records and boundary channels are consumed by the
v5.1 structured executor. The checkpoint is not retrained with v5.1
$C_A-C_0$ labels or coalition-delta supervision, and no parameter update uses
feedback from a formal evaluation run. Model parameters remain fixed during
the formal runs.

The **analytic oracle** mode removes the learned checkpoint and uses the
analytic exact-pressure/ranking scaffold through the same executor interface.
It is included only to diagnose whether the coordination mechanism can exploit
informative conflict and opportunity signals. It is not a global MWIS upper
bound, not privileged future-solution access, and not a substitute for the
learned-policy evaluation. Learned and analytic results must therefore be
reported in separately labelled tables.

## 9. Ablation definitions

Let factor $A$ denote the registered semantic message pathway and factor
$B$ denote admission of the structured coalition closure to the exact
candidate union.

| Configuration | Semantic pathway A | Coalition closure B | Purpose |
|---|---:|---:|---|
| `Full` | real combined channels | on | complete learned/analytic method |
| `FrozenAnchorComm` | real channels | off | $A1B0$ factorial cell |
| `SelectionOnly` | zero channels | on | $A0B1$ factorial cell and protected $C_0$ coalition |
| `FrozenAnchorSelectionOnly` | zero channels | off | $A0B0$ common exact-repair baseline |
| `AlternatingConflictBenefit` | odd conflict / even benefit | on | temporally separated conflict--reward semantics |
| `OneShotComm` | real once, then zero | on | persistence/frequency control |
| `ShuffledComm` | size-preserving misassigned channels | on | semantic-correspondence control |
| `NoPeerFinalShield` | no peer channels | off | information-barrier baseline with the same global safety shield |

The primary factorial contrasts are $A\mid B$, $A\mid\neg B$,
$B\mid A$, $B\mid\neg A$, and the $A\times B$ interaction. Protocol
contrasts include `Full` minus `OneShotComm`, `Full` minus `ShuffledComm`, and
`Full` minus `NoPeerFinalShield`. All contrasts must be computed seed by seed
under the same initial schedule and paid-query budget.

## 10. Paper contribution statement

The method makes two substantive algorithmic contributions.

1. **Structured tri-graph primal--dual messaging.** Instead of communicating a
   second compressed summary after graph partitioning, agents exchange bounded
   packets of traceable boundary candidates and response-relevant local
   structure. Private subgraphs generate executable responses, the dynamic
   boundary graph prices and coordinates cross-agent conflicts, and the public
   global graph supplies residual context and an authoritative safety shield.

2. **Counterfactual semantic safeguarding with coalitional
   delete--replace--refill.** The coordinator constructs a new feasible
   multi-agent response rather than merely moving a future search anchor. It
   evaluates same-state zero and treated semantic branches, diversifies their
   microresponses through deterministic counterfactual feedback, protects the
   zero-semantic coalition candidate in an exact union, and prohibits rejected
   semantic records from contaminating later search domains.

Together, these mechanisms turn communication from descriptive advice into a
verifiable joint action while preserving equal local-worker budgets and a
monotone committed objective. The appropriate empirical claim is therefore
about the contribution of structured multi-agent coordination under a fixed
worker, not about replacing the local worker or proving global optimality.

# Agent Teams — Master Reference

Source: <https://code.claude.com/docs/en/agent-teams> (plus `sub-agents`, `hooks`,
`tools-reference`). Captured **2026-08-28**, documenting behaviour as of Claude Code
**v2.1.178+**, with version-gated notes up to **v2.1.246**.

Agent teams are **experimental**. Behaviour changes between patch versions — the
version notes in this guide are load-bearing, not trivia. Re-verify against the live
docs before relying on a detail for something expensive.

---

## 1. The mental model

One session is the **lead**. It spawns **teammates** — each a full, independent Claude
Code session with its own context window. Teammates talk *to each other directly*, not
just back to the lead. You can open any teammate's transcript and message it yourself.

```
   SUBAGENTS                          AGENT TEAMS
   ─────────                          ───────────
   main ──> sub ──> result            lead ⇄ mate A
   main ──> sub ──> result             ↕  ⤫   ↕
   (results flow back up)             mate B ⇄ mate C
                                      (+ shared task list, mailboxes)
```

The single most important structural difference:

> **A subagent returns its output to the caller. A teammate does not.**
> When a teammate goes idle the lead gets a notification that it *stopped* — the
> notification carries **no output**. A teammate shares results only by messaging
> someone or by updating the shared task list.

Every orchestration bug in section 10 traces back to that one sentence.

---

## 2. Choosing the right tool

| Need | Use |
| :--- | :--- |
| Focused worker, only the result matters | **Subagent** |
| Workers must debate, challenge, coordinate | **Agent team** |
| Pass findings between sessions *you* started | **Cross-session messaging** |
| Fully manual parallelism, isolated checkouts | **Git worktrees** |

| | Subagents | Agent teams |
| :--- | :--- | :--- |
| **Context** | Own window; result returns to caller | Own window; fully independent |
| **Communication** | Return a result. Named subagents can message each other | Teammates message each other directly |
| **Coordination** | Main agent manages all work | Self-coordination + shared task list |
| **Best for** | Focused tasks | Work needing discussion |
| **Token cost** | Lower (summarised back) | **Higher** (each mate is a full instance) |

**Teams earn their cost on:** research and review, new modules/features with clean
ownership boundaries, debugging with competing hypotheses, cross-layer work
(frontend/backend/tests each owned by one mate).

**Teams lose to a single session on:** sequential work, same-file edits, anything with
heavy dependencies between steps.

---

## 3. Enabling

```json
// .claude/settings.local.json  (this project — already set)
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

Settings-file `env` values are **reapplied to the running session on save**, and the
variable is re-read every time Claude spawns a subagent. Flipping it to `"0"` takes
effect on the next spawn — no restart needed.

Precedence, later wins: user settings → project settings → local settings →
`--settings` payload → **managed settings**. A `"0"` in user settings is overridden by a
`"1"` in any of the higher ones.

### Two hard preconditions

1. **Interactive only.** In `-p` / headless mode, including Agent SDK sessions, Claude
   never spawns teammates. A named subagent runs as an ordinary subagent instead.
2. **Enabling changes ordinary delegation.** While teams are on, *any* subagent Claude
   names launches as a **teammate** — and Claude names subagents on its own initiative.
   Teams can form when you never asked for one. See §10.1.

---

## 4. ⚠️ Task tools on Opus 5 — read this before designing any team

Per `tools-reference#task-tool-availability`, in **v2.1.233+** these tools are **absent
by default** on Opus 4.8, Sonnet 5, Fable 5, Mythos 5, and later versions of those
families: `TodoWrite`, `TaskCreate`, `TaskGet`, `TaskUpdate`, `TaskList`.

**This project runs Opus 5, so the shared task list is OFF by default.** (Verified
empirically: no `Task*` tools appear in this session's tool list.)

Consequences for team design here:

- No shared task list. **Teammates coordinate purely through messages.**
- Task dependencies, self-claiming, file-locked claim arbitration — all unavailable.
- The `TaskCreated` hook **never fires**. `TaskCompleted` fires only via the teammate
  turn-end path, not via `TaskUpdate`.
- The lead must hand-assign work in spawn prompts and reconcile results by message.

To turn the task list on:

```bash
CLAUDE_CODE_ENABLE_TODO_TOOLS=1 claude       # every model, every provider
claude --allowedTools TaskCreate             # or name one tool
claude --tools ...                           # or restrict the built-in set
```

Two asymmetries worth remembering:

- An **in-process** teammate inherits the *session's* tool availability.
  A **split-pane** teammate is a separate process, so **its own model decides**.
- Background sessions and Claude Code on the web always provide these tools.

**Design rule:** unless you have exported `CLAUDE_CODE_ENABLE_TODO_TOOLS=1`, write team
prompts that assume message-based coordination. Do not build a plan around task
dependencies that will not exist.

---

## 5. Spawning a team

Ask in natural language. Claude spawns teammates by calling the Agent tool **with a
`name`** while teams are enabled — and **does not ask you to confirm**.

```text
I'm designing a CLI tool that tracks TODO comments across a codebase.
Spawn three teammates to explore this from different angles: one on UX,
one on technical architecture, one playing devil's advocate.
```

Three roles, mutually independent, none blocking another — that is what makes this
prompt work.

Claude may use subagents instead. **The agent panel shows both, so the panel alone does
not prove a team formed.** If you got subagents, ask again and say "agent team"
explicitly.

**Name your teammates in the spawn prompt.** The lead assigns names anyway, but naming
them yourself gives you stable addresses to reference in later turns.

### The agent panel (in-process)

| Key | Action |
| :--- | :--- |
| ↑ / ↓ | Select a teammate |
| Enter | Open its transcript and message it directly |
| Esc | Clear selection; while viewing, interrupts that teammate's turn |
| `x` | Stop the selected teammate |
| Ctrl+T | Toggle the task list |

Idle-row behaviour (**v2.1.199+**): an idle row stays visible while *any* agent is still
working. Once everything is idle, idle rows hide after 30s and return on the teammate's
next turn. **A hidden row means hidden, not stopped** — the teammate is still running and
addressable by name. More than three idle at once collapse into one `N idle agents` row;
Enter expands. (v2.1.181–v2.1.198 hid each row 30s after its own turn; before v2.1.181
nothing hid.)

---

## 6. Models, effort, and display mode

### Model precedence — first match wins

1. `CLAUDE_CODE_SUBAGENT_MODEL`, when set to anything other than `inherit`
2. The model your spawn prompt names for that teammate
3. For an **in-process** teammate from a subagent definition: the definition's `model`
4. The lead's current model

`teammateDefaultModel` was **removed in v2.1.234** and a leftover value is ignored.

If your org's `availableModels` allowlist blocks the requested model: a **family alias**
(`opus`) resolves to the newest permitted version of that family on the Anthropic API and
Claude Platform on AWS; anything else blocked — including aliases on providers with
provider-specific IDs — falls back to **the lead's model**.

**Effort** is inherited from the lead (in split panes, from v2.1.186). `/effort` still
applies to a viewed teammate's later turns. **Model and fast mode are frozen at spawn** —
`/model` and `/fast` only ever change the lead (v2.1.199+ shows a notice saying so).

### Display mode

| Mode | Behaviour |
| :--- | :--- |
| `in-process` | **Default.** All teammates in the main terminal. Works anywhere. |
| `auto` | Split panes if already in tmux, or iTerm2 with `it2`; else in-process |
| `tmux` | Split panes, auto-detecting tmux vs iTerm2 |
| `iterm2` | **v2.1.186+.** iTerm2 native panes; requires the `it2` CLI |

```json
// ~/.claude/settings.json
{ "teammateMode": "auto" }
```

```bash
claude --teammate-mode auto     # experimental, absent from --help
```

Before v2.1.179 the default was `"auto"`, so upgraded sessions that used to open split
panes now stay in one terminal unless set explicitly.

Split panes need tmux or iTerm2 + [`it2`](https://github.com/mkusaka/it2) (plus
**iTerm2 → Settings → General → Magic → Enable Python API**). tmux works best on macOS;
`tmux -CC` in iTerm2 is the suggested entrypoint. **Not supported** in VS Code's
integrated terminal, Windows Terminal, or Ghostty.

---

## 7. Reusable roles: subagent definitions as teammates

Define a role once in `.claude/agents/` or `~/.claude/agents/`, use it as both a subagent
and a teammate:

```text
Spawn a teammate using the security-reviewer agent type to audit the auth module.
```

```markdown
---
name: security-reviewer
description: Audits code for security vulnerabilities
tools: Read, Glob, Grep, Bash
model: sonnet
---

You are a security reviewer. Report findings with severity ratings and file:line
evidence. Never edit code.
```

Frontmatter fields (only `name` and `description` are required): `tools`,
`disallowedTools`, `model`, `permissionMode`, `maxTurns`, `skills`, `mcpServers`,
`hooks`, `memory`, `background`, `effort`, `isolation`, `color`, `initialPrompt`.

Scope precedence, highest first: managed settings → `--agents` CLI flag →
`.claude/agents/` → `~/.claude/agents/` → plugin `agents/`.

### ⚠️ In-process and split-pane teammates apply the definition *differently*

| Field | In-process teammate | Split-pane teammate |
| :--- | :--- | :--- |
| `tools` | Limited to the list, **plus `SendMessage`** (plus the four `Task*` tools if the session has them) | Limited to the list |
| `model` | Used, if neither env var nor prompt names one | **Ignored** |
| **Body** | **Appended** to the default system prompt | **Replaces** the default system prompt |
| `skills` | **Ignored** — loads from project/user settings | **Ignored** — same |
| `mcpServers` | **Ignored** — loads from project/user settings | **Applied** |

The body row is the sharp edge: the *same* definition yields an augmented agent
in-process and a from-scratch agent in a split pane. Write role bodies that stand alone
if you might use either mode.

Claude Code watches the `agents/` directories and picks up edits within seconds. Restart
is needed only when: you created a scope's **first** agent file in a brand-new directory,
the file lives under an `--add-dir` path, or the session ran with
`--disable-slash-commands`.

A file is **silently skipped** (debug log only — run `--debug`) when it has no `name`, a
`name` starting with `-` or containing `:`, a `name` without a `description`, or YAML
that does not parse.

---

## 8. Architecture

| Component | Role |
| :--- | :--- |
| **Team lead** | Main session; spawns teammates, coordinates, synthesises |
| **Teammates** | Separate Claude Code instances working assigned tasks |
| **Task list** | Shared work items teammates claim and complete |
| **Mailbox** | Per-agent messaging |

Everything is keyed to a **session-derived name**: `session-` + the first eight
characters of the session ID.

```
~/.claude/teams/{team-name}/config.json              # removed when session ends
~/.claude/teams/{team-name}/inboxes/{agent}.json     # mailboxes
~/.claude/tasks/{team-name}/                         # PERSISTS across resume
```

- **Never hand-edit or pre-author `config.json`.** It holds live runtime state (session
  IDs, tmux pane IDs) and is overwritten on the next state update.
- There is **no project-level equivalent**. A `.claude/teams/teams.json` in your repo is
  just an ordinary file, not configuration.
- The task directory is never uploaded; retention follows `cleanupPeriodDays`.
- `config.json` has a `members` array (name + agent ID). The lead's entry always carries
  agent type `team-lead`; a teammate carries whatever type it was spawned with, or omits
  the field. **Teammates can read this file to discover each other.**

**Mailbox robustness (v2.1.207+):** every entry is validated on read; malformed entries
are reported as errors and removed, and valid messages still get delivered. Earlier
versions threw an error every second and blocked that mailbox until you deleted the file
by hand. A message counts as *sent* only when the write to the recipient's mailbox
succeeds — on failure the sender gets an error and nothing is delivered.

Cleanup is automatic on session exit. There is no manual teardown step; `TeamCreate` and
`TeamDelete` no longer exist, and the `team_name` input on the Agent tool is accepted but
ignored.

---

## 9. Permissions and the trust boundary

- Teammates **start with the lead's permission settings**. `--dangerously-skip-permissions`
  on the lead means every teammate runs that way too.
- You can change an individual teammate's mode after spawn, but **not per-teammate modes
  at spawn time**.
- **Teammate permission prompts surface in the lead session — you approve them there.**

### Plan approval is the one designed exception

A teammate spawned while the lead is in **plan mode** works read-only until its plan is
ready, then sends a plan approval request. **Claude Code approves it in the lead's session
automatically, without the lead reviewing it.** Its subsequent edits and commands still hit
normal permission prompts.

```text
# switch the lead into plan mode first, then:
Spawn an architect teammate to refactor the authentication module.
```

### Inter-agent messages are untrusted input

When one agent messages another, the receiver is told **the message came from another
Claude session, not from you.**

- A teammate **cannot** approve a permission prompt or give consent on your behalf.
- A teammate denied an action **cannot** relay it to another teammate to get around the
  check.
- In **auto mode**, the classifier treats a relayed approval claim as untrusted input, and
  reviews *every* message — plain or structured (shutdown requests, plan approval
  responses) — before delivery. A blocked message never reaches the recipient.

The same rules apply to messages arriving from your other Claude Code sessions.

---

## 10. Context and communication

A teammate loads the same project context as a regular session — **CLAUDE.md, MCP
servers, and skills** — plus the spawn prompt. **The lead's conversation history does not
carry over.**

- **Automatic delivery**: messages are pushed to recipients; the lead never polls.
- **Idle notifications**: fired when a teammate stops, and they **carry no output**.
  (v2.1.198+: a turn ending on an API error notifies the lead *as a failure* with the
  error text, instead of looking like a normal finish.)
- **Messaging**: address one teammate by name. **To reach everyone, send one message per
  recipient** — there is no broadcast.
- A message from the lead or another teammate **wakes an in-process teammate that is
  waiting to retry a failed API request**, so it retries immediately.

### Token cost

Each teammate is a full context window; cost scales with active teammates. Worth it for
research, review, and new features — not for routine work.

**Cache gotcha:** an in-process teammate's requests fall outside the main conversation's
cache TTL bucket, so its cache holds **five minutes** by default, even on a subscription.
Set `subagentPromptCacheTtl` to `1h` to extend it — the API bills 1-hour cache writes at
a higher rate.

---

## 11. Quality gates with hooks

Three hook events cover the team lifecycle. **None support matchers**; all fire on every
occurrence.

| Hook | Fires | Exit code 2 does |
| :--- | :--- | :--- |
| `TeammateIdle` | Teammate about to go idle | Feeds stderr back; teammate keeps working |
| `TaskCreated` | Task being created via `TaskCreate` | Blocks creation, returns stderr as the tool error |
| `TaskCompleted` | Task marked complete, **or** teammate ends its turn with in-progress tasks | Blocks completion, feeds stderr back |

Payload fields: `TeammateIdle` gets `teammate_name` and `team_name`. `TaskCreated` and
`TaskCompleted` get `task_id`, `task_subject`, and optionally `task_description`,
`teammate_name`, `team_name`. **`team_name` is deprecated** in all three.

JSON decision control differs per event:

- `TeammateIdle`: `{"continue": false, "stopReason": "..."}` stops the teammate entirely.
- `TaskCreated`: `{"decision": "block", "reason": "..."}` blocks; `continue: false` is
  **ignored**.
- `TaskCompleted`: `continue: false` stops the teammate **only** on the turn-end path;
  on the `TaskUpdate` path it is ignored, though exit code 2 still blocks.

```bash
#!/bin/bash
# TeammateIdle — refuse to let a teammate stop with a missing build artifact
if [ ! -f "./dist/output.js" ]; then
  echo "Build artifact missing. Run the build before stopping." >&2
  exit 2
fi
exit 0
```

```bash
#!/bin/bash
# TaskCompleted — no green tests, no closed task
INPUT=$(cat)
TASK_SUBJECT=$(echo "$INPUT" | jq -r '.task_subject')
if ! npm test 2>&1; then
  echo "Tests not passing. Fix before completing: $TASK_SUBJECT" >&2
  exit 2
fi
exit 0
```

> **In this project (Opus 5, no Task tools):** `TaskCreated` never fires and
> `TaskCompleted` only fires on the teammate turn-end path. **`TeammateIdle` is your only
> reliable gate.** Put the real verification there.

---

## 12. Best practices

**Give teammates enough context.** They inherit project context but *not* the lead's
history. Put the specifics in the spawn prompt:

```text
Spawn a security reviewer teammate with the prompt: "Review the authentication
module at src/auth/ for security vulnerabilities. Focus on token handling,
session management, and input validation. The app uses JWT tokens stored in
httpOnly cookies. Report any issues with severity ratings."
```

**Team size: start with 3–5.** Token cost scales linearly, coordination overhead scales
worse, and returns diminish. *15 independent tasks is still a good fit for 3 teammates.*
**Three focused teammates beat five scattered ones.**

**Task size.** Too small → coordination costs more than it saves. Too large → long
unsupervised stretches and wasted work. Right → a self-contained unit with a clear
deliverable: one function, one test file, one review. Aim for **5–6 tasks per teammate**
so the lead can reassign when someone stalls.

**Avoid file conflicts.** Two teammates editing one file means overwrites. Partition by
file ownership, not by topic.

**Make the lead wait.** Leads sometimes start implementing instead of delegating:

```text
Wait for your teammates to complete their tasks before proceeding
```

**Monitor and steer.** Check transcripts, redirect bad approaches, synthesise as findings
arrive. Unattended teams waste tokens confidently.

**Start with read-only work.** Your first teams should review a PR, research a library,
or investigate a bug — parallel exploration's upside without parallel-write coordination.

---

## 13. Playbooks

### Parallel code review

One reviewer fixates on one class of issue. Assign disjoint lenses:

```text
Spawn three teammates to review PR #142:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review and report findings.
```

Same input, three filters, lead synthesises.

### Competing hypotheses (the strongest pattern)

Sequential investigation anchors: once one theory is explored, everything after is biased
toward it. Adversarial parallelism beats that.

```text
Users report the app exits after one message instead of staying connected.
Spawn 5 agent teammates to investigate different hypotheses. Have them talk to
each other to try to disprove each other's theories, like a scientific debate.
Update the findings doc with whatever consensus emerges.
```

The debate structure *is* the mechanism — the theory that survives active attempts at
falsification is far more likely to be the real root cause. Note the shared artifact
("findings doc"): with no task list, a file is how consensus gets recorded.

### Cross-layer feature

One teammate per layer, disjoint file ownership, one integrator:

```text
Spawn 3 teammates for the export feature, each owning separate files:
- backend: src/api/export/* only
- frontend: src/ui/export/* only
- tests: tests/export/* only
Each reports to me by message when done. Do not edit outside your own directory.
```

---

## 14. Failure modes

### 14.1 Claude spawns teammates when you wanted subagents

The most common and most damaging failure. While teams are enabled, any subagent Claude
*names* becomes a teammate — and Claude names them on its own. **An orchestration flow
that waits on subagent results will stall**, because a teammate's idle notification
carries no output.

Fix — set the variable to `"0"`; no restart needed:

```json
{ "env": { "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "0" } }
```

Claude may still name subagents afterwards, and the name still works as a `SendMessage`
address. Remember managed settings and higher-precedence files can re-enable it.

### 14.2 Teammates not appearing

Check, in order: are they hidden rather than stopped (§5)? Was the task complex enough to
warrant a team? For split panes, `which tmux`; for iTerm2, is `it2` installed and the
Python API enabled? Message a teammate by name to bring a hidden row back.

### 14.3 Too many permission prompts

Every teammate's prompt lands on the lead. **Pre-approve common operations in your
permission settings before spawning.**

### 14.4 Agents stopping early

Teammates stop on errors instead of recovering. Open the transcript, then either give
direct instructions or spawn a replacement. **The lead also stops early**, declaring
victory before the work is done — tell it to keep going.

### 14.5 Orphaned tmux sessions

```bash
tmux ls
tmux kill-session -t <session-name>
```

---

## 15. Limitations

- **No session resumption with in-process teammates.** `/resume` and `/rewind` do not
  restore them; the lead may try to message teammates that no longer exist. Tell it to
  spawn new ones.
- **Task status lags.** Teammates fail to mark tasks complete, blocking dependents. Check
  whether the work is actually done and fix the status by hand.
- **Shutdown is slow.** A teammate finishes its current request or tool call first.
- **One team per session**, scoped to that session. No named teams, no sharing.
- **No nested teams.** Only the lead manages the team.
- **No background subagents from in-process teammates.** A teammate's background work
  cannot outlive the lead's process. A subagent definition with `background: true` errors;
  `run_in_background: true` fails or silently runs in the foreground.
- **Lead is fixed** for the session's lifetime. No promotion, no transfer.
- **Permissions set at spawn** (§9).
- **Split panes need tmux or iTerm2.**

To shut a teammate down gracefully, name it — the lead sends a request the teammate can
approve or reject with an explanation:

```text
Ask the researcher teammate to shut down
```

---

## 16. Pre-flight checklist

A paired do/don't card, grounded in this project's own failures, lives in
[`agent-teams-lilly.md`](agent-teams-lilly.md). Single-sourced there — do not copy it here.

Before spawning a team:

- [ ] Is this genuinely parallel? Sequential or same-file work → single session.
- [ ] Interactive session? Headless never spawns teammates.
- [ ] Do I need the shared task list? On Opus 5 it is **off** — export
      `CLAUDE_CODE_ENABLE_TODO_TOOLS=1` or design for messages (§4).
- [ ] 3–5 teammates, each with a distinct lens and **disjoint file ownership**.
- [ ] Does each spawn prompt carry its own context? History does not transfer.
- [ ] Named teammates, so I can address them later.
- [ ] Common operations pre-approved, so prompts do not pile up on the lead.
- [ ] How does each teammate **report**? Idle notifications carry nothing — require an
      explicit message to the lead or a write to a shared file.
- [ ] `TeammateIdle` hook in place if quality must be enforced rather than requested.
- [ ] Am I going to monitor this, or leave it running unattended?

---

## 17. Quick reference

```bash
CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1   # enable teams ("0" disables; live-reloaded)
CLAUDE_CODE_ENABLE_TODO_TOOLS=1          # restore Task tools on Opus 5 / Sonnet 5 / …
CLAUDE_CODE_SUBAGENT_MODEL=sonnet        # highest-precedence teammate model
claude --teammate-mode auto              # split panes when available
```

```
~/.claude/teams/session-XXXXXXXX/config.json           # runtime state — do not edit
~/.claude/teams/session-XXXXXXXX/inboxes/{agent}.json  # mailboxes
~/.claude/tasks/session-XXXXXXXX/                      # persists across resume
.claude/agents/*.md                                    # reusable teammate roles
```

Settings: `teammateMode`, `subagentPromptCacheTtl`, `cleanupPeriodDays`.
Hooks: `TeammateIdle`, `TaskCreated`, `TaskCompleted`.
Removed: `TeamCreate`, `TeamDelete`, `teammateDefaultModel` (v2.1.234).
Deprecated: `team_name` (Agent tool input, and all three hook payloads).

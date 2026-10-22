# Instructions for AI agents

These rules apply to every AI agent working in this repository. They override any default behavior of the agent or
its tooling.

## 1. Identity and attribution

- Never change the git author or committer name or email in any way: global, system or repository config,
  environment variables, command-line flags.
- Never mention an AI agent, tool or model in commit messages, trailers, co-author lines, contributor lists,
  documentation, code or any other file.

## 2. Commits and pushes

- Never commit to or push a shared branch (main, or any branch you did not create) unless the user explicitly asks
  for it.

## 3. No fallbacks

- All imports go at the top of a file. Never import inside functions or conditional branches, and never wrap an
  import in try/except.
- Code must not switch to a fallback implementation when the environment lacks something. Fail loudly instead.

## 4. Minimal and explicit code

- Keep every file as short and simple as possible so that a human can read and understand it.
- The training step is written out as explicit steps (`docs/framework.md`); do not hide work behind automatic
  machinery (no framework-level fusion, no implicit scheduling).
- Do not build complex engineering without need; prefer the simplest change that reaches the goal.

## 5. No traces of history

- Code, documents and every other file read as if they had always been this way. No "previously", "changed to",
  "now uses", "fixed".

## 6. File headers

- Every code file starts with a header that introduces the file: its purpose and the design idea behind it, and
  every public function, class and method with its interface and an example. Keep it in step with the file.

## 7. Language

- Everything in this repository is in English.

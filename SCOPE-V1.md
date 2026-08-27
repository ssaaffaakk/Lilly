# v1 — what ships, and what does not

Written because every finding opened a branch and no branch was closing. The
work was good and none of it was wasted; what was missing was a line saying
where v1 ends. Without that line every new discovery joins v1 by default, and
v1 never arrives.

**Rule: anything found after this file was written goes to v2. No exceptions.**

## In v1 — nothing publishes until all five are done

1. **Pick the winning arm against the pre-registered threshold.** — done. Both
   arms cleared; the tie-break picked the higher chrF2. `training/PREREGISTRATION.md`.
2. **Build it, and bind the build to its numbers.** — done. The report names the
   build by content hash and the publisher refuses any other.
3. **An honest model card.** — done. Scores from the measured build, the
   benchmark result that does not support the project's own pitch, and the
   limits, all in `models/lilly/README.md`.
4. **Run the app end to end, once.** — done. Text, a photograph of Bosnian
   letters, a photograph of dates, 413, 400, and speech, all against the running
   server.
5. **Upload to Hugging Face.** — waiting on the owner. The dry run passes:
   899.9 MB, 25 files, every folder either published or explained.

## In v2 — real work, and none of it blocks v1

- **BosnianBench is built and it does not support the claim** (91.5% → 91.7%,
  p = 0.465). It stays in v2 not because it is unfinished but because what it
  needs next — Turkisms that no news corpus contains, a base that is not already
  trained across South Slavic — is a research problem, not a release task. The
  card says so plainly, which is more honest than holding the release until the
  number improves.
- Reader and detector improvements. The recogniser is fine-tuned; the CRAFT
  detector is stock, and nothing here measures finding text on a real photograph.
- Real photographs. Everything is synthesised.
- Further arms, hyperparameter search, decoding-parameter sweeps.
- Hosting **was** in this list as "a decision about money, not a task", on the
  grounds that Lilly needs about 4 GB of RAM and Render's free tier gives 512 MB.
  That was a conclusion reached without checking the alternatives. Hugging Face
  Spaces' CPU Basic tier is 2 vCPU and **16 GB of RAM at no cost** — read off
  huggingface.co/pricing, not assumed — which is five times the 3.2 GB peak
  measured on a photo request, and it sits on the same platform the weights were
  just published to, so the app loads them without going over the network.

  So it is a task, not a bill, and it belongs in whatever ships next rather than
  in a someday list. Recorded here rather than quietly corrected because the
  mistake is worth keeping: a line written from memory rather than measurement,
  in a file whose whole purpose is to stop exactly that. Caught by the Egitim
  session, which also insisted the correction be verified rather than believed.

## What v1 does not claim

That Lilly understands Bosnian better than the base model. It does not, by the
only measure built to test it. What it does is translate a little better on
FLORES, never print a language tag into its own output, read photographs far
better than the stock recogniser, and hear Bosnian speech with three points less
error. Those are the claims, and each has a number behind it.

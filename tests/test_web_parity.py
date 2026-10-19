"""The web demo's JavaScript port must give exactly the same answers as the Python package.

Runs web/dna_align.js in Node on many random inputs and compares matrices, scores,
alignments, optimal-alignment counts, statistics and co-optimal enumeration.
Skipped automatically when Node.js is not installed."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from dna_alignment import (ScoringScheme, compute_stats, enumerate_global_alignments,
                           needleman_wunsch, random_dna, smith_waterman, validate_scoring)

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")

RUNNER = """
const DNA = require(process.argv[1]);
const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = cases.map(c => {
  const sc = {match: c.sc[0], mismatch: c.sc[1], gap: c.sc[2]};
  const nw = DNA.needlemanWunsch(c.s1, c.s2, sc), sw = DNA.smithWaterman(c.s1, c.s2, sc);
  const st = DNA.stats(nw);
  return {nwF: nw.F, nwScore: nw.score, nwA: [nw.a1, nw.a2], count: nw.optimalCount,
          nwPath: nw.path, enumd: DNA.enumerateGlobal(nw, 50),
          swF: sw.F, swScore: sw.score, swA: [sw.a1, sw.a2], swEnd: sw.end,
          stats: [st.matches, st.mismatches, st.gaps, st.length, st.identity, st.rescored],
          val: DNA.validateScoring(c.sc[0], c.sc[1], c.sc[2]).errors.length};
});
process.stdout.write(JSON.stringify(out));
"""


def run_js(cases):
    proc = subprocess.run(["node", "-e", RUNNER, str(ROOT / "web" / "dna_align.js")],
                          input=json.dumps(cases), capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def test_js_matches_python():
    rng = random.Random(42)
    schemes = [(2, -1, -2), (1, -1, -1), (1, -3, -2), (3, 0, -1), (2, -1, 0), (5, -4, -10)]
    cases = [{"s1": "AGTCA", "s2": "AGCA", "sc": [2, -1, -2]},
             {"s1": "", "s2": "", "sc": [2, -1, -2]},
             {"s1": "ACG", "s2": "", "sc": [2, -1, -2]},
             {"s1": "ATTA", "s2": "ATA", "sc": [2, -1, -2]},
             {"s1": "AAAA", "s2": "TTTT", "sc": [2, -1, -2]}]
    for k in range(150):
        cases.append({"s1": random_dna(rng.randint(0, 25), rng),
                      "s2": random_dna(rng.randint(0, 25), rng),
                      "sc": list(schemes[k % len(schemes)])})
    js = run_js(cases)
    for c, o in zip(cases, js):
        sc = ScoringScheme(*c["sc"])
        nw, sw = needleman_wunsch(c["s1"], c["s2"], sc), smith_waterman(c["s1"], c["s2"], sc)
        assert o["nwF"] == nw.matrix and o["nwScore"] == nw.score
        assert o["nwA"] == [nw.aligned1, nw.aligned2]
        assert o["count"] == nw.optimal_count
        assert [tuple(p) for p in o["nwPath"]] == nw.path
        assert [tuple(x) for x in o["enumd"]] == enumerate_global_alignments(nw, 50)
        assert o["swF"] == sw.matrix and o["swScore"] == sw.score
        assert o["swA"] == [sw.aligned1, sw.aligned2] and tuple(o["swEnd"]) == sw.end
        st = compute_stats(nw.aligned1, nw.aligned2, sc, nw.score)
        assert o["stats"] == [st.matches, st.mismatches, st.gaps, st.length,
                              st.identity, st.recomputed_score]
        assert o["val"] == len(validate_scoring(*c["sc"])[0])

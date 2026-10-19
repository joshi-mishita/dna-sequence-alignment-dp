/* DNA alignment core for the web demo — a line-for-line port of dna_alignment/alignment.py.
 * Same conventions: seq1 = rows (i), seq2 = columns (j); tie rule DIAG > UP > LEFT;
 * Smith-Waterman traceback starts at the first maximal cell in row-major order.
 * tests/test_web_parity.py runs this file in Node and compares it with the Python results. */
(function (root) {
  "use strict";
  var D = "D", U = "U", L = "L", PREF = [D, U, L];
  var GLOBAL = "Needleman-Wunsch", LOCAL = "Smith-Waterman";

  function clean(raw) { return String(raw || "").replace(/\s+/g, "").toUpperCase(); }

  function validateSequence(raw, name, maxLen) {
    maxLen = maxLen || 500;
    var seq = clean(raw), errors = [], bad = {}, pos = [];
    for (var k = 0; k < seq.length; k++) {
      if ("ACGT".indexOf(seq[k]) < 0) { bad[seq[k]] = 1; if (pos.length < 5) pos.push(k + 1); }
    }
    var badList = Object.keys(bad).sort();
    if (badList.length) errors.push(name + " contains invalid character(s) " +
      badList.map(function (c) { return "'" + c + "'"; }).join(", ") + " at position(s) " +
      pos.join(", ") + (pos.length === 5 ? " …" : "") + ". Only A, T, G and C are allowed.");
    if (seq.length > maxLen) errors.push(name + " is " + seq.length + " bases long; the maximum is " + maxLen + ".");
    return { sequence: seq, errors: errors, ok: errors.length === 0 };
  }

  function validateScoring(match, mismatch, gap) {
    var errors = [], warnings = [], vals = { "Match score": match, "Mismatch score": mismatch, "Gap penalty": gap };
    Object.keys(vals).forEach(function (k) {
      var v = vals[k];
      if (typeof v !== "number" || !isFinite(v) || Math.floor(v) !== v) errors.push(k + " must be a whole number.");
      else if (Math.abs(v) > 100) errors.push(k + " must be between -100 and 100.");
    });
    if (errors.length) return { errors: errors, warnings: warnings };
    if (match <= mismatch) errors.push("Match score must be greater than the mismatch score; otherwise the algorithm has no reason to prefer identical bases.");
    if (gap > 0) errors.push("Gap penalty must be zero or negative; a positive gap score would reward inserting gaps.");
    if (match <= 0) warnings.push("Match score is not positive, so alignment scores will never increase.");
    if (mismatch > 0) warnings.push("Mismatch score is positive, so mismatches are rewarded.");
    if (gap === 0) warnings.push("Gap penalty is zero, so gaps are free and alignments may contain many gaps.");
    if (gap < 0 && 2 * gap > mismatch) warnings.push("Two gaps cost less than one mismatch, so the algorithm may prefer an insertion + deletion pair over a substitution.");
    return { errors: errors, warnings: warnings };
  }

  function fill(s1, s2, sc, local) {
    var m = s1.length, n = s2.length, g = sc.gap, F = [], P = [], steps = [], i, j;
    for (i = 0; i <= m; i++) { F.push(new Array(n + 1).fill(0)); P.push(new Array(n + 1).fill("")); }
    for (i = 1; i <= m; i++) { F[i][0] = local ? 0 : i * g; P[i][0] = local ? "" : U; }
    for (j = 1; j <= n; j++) { F[0][j] = local ? 0 : j * g; P[0][j] = local ? "" : L; }
    for (i = 1; i <= m; i++) {
      for (j = 1; j <= n; j++) {
        var a = s1[i - 1], b = s2[j - 1], s = a === b ? sc.match : sc.mismatch;
        var c = { D: F[i - 1][j - 1] + s, U: F[i - 1][j] + g, L: F[i][j - 1] + g };
        var best = Math.max(c.D, c.U, c.L), chosen = "", zero = false;
        if (local && best <= 0) { best = 0; zero = true; }
        else chosen = PREF.filter(function (d) { return c[d] === best; }).join("");
        F[i][j] = best; P[i][j] = chosen;
        steps.push({ i: i, j: j, a: a, b: b, s: s, diag: c.D, up: c.U, left: c.L, best: best, chosen: chosen, zero: zero });
      }
    }
    return { F: F, P: P, steps: steps };
  }

  function traceback(s1, s2, F, P, end, local) {
    var i = end[0], j = end[1], c1 = [], c2 = [], path = [[i, j]], steps = [];
    for (;;) {
      var opts = P[i][j];
      var stop = local ? (F[i][j] === 0 || !opts) : (i === 0 && j === 0);
      if (stop) { steps.push({ i: i, j: j, score: F[i][j], move: null, options: opts, p1: c1.slice().reverse().join(""), p2: c2.slice().reverse().join("") }); break; }
      var mv = opts[0], a, b, ni, nj;
      if (mv === D) { a = s1[i - 1]; b = s2[j - 1]; ni = i - 1; nj = j - 1; }
      else if (mv === U) { a = s1[i - 1]; b = "-"; ni = i - 1; nj = j; }
      else { a = "-"; b = s2[j - 1]; ni = i; nj = j - 1; }
      c1.push(a); c2.push(b);
      steps.push({ i: i, j: j, score: F[i][j], move: mv, options: opts, col1: a, col2: b, ni: ni, nj: nj,
        p1: c1.slice().reverse().join(""), p2: c2.slice().reverse().join("") });
      i = ni; j = nj; path.push([i, j]);
    }
    path.reverse();
    return { a1: c1.reverse().join(""), a2: c2.reverse().join(""), path: path, steps: steps };
  }

  function countPaths(P, m, n) {
    var C = [], i, j;
    for (i = 0; i <= m; i++) C.push(new Array(n + 1).fill(0));
    C[0][0] = 1;
    for (i = 0; i <= m; i++) for (j = 0; j <= n; j++) {
      if (!i && !j) continue;
      var t = 0, p = P[i][j];
      for (var k = 0; k < p.length; k++) t += p[k] === D ? C[i - 1][j - 1] : p[k] === U ? C[i - 1][j] : C[i][j - 1];
      C[i][j] = t;
    }
    return C[m][n];  // Number; exact up to 2^53, shown as "many" beyond that
  }

  function enumerateGlobal(r, limit) {
    var out = [], s1 = r.seq1, s2 = r.seq2, P = r.P;
    var stack = [[s1.length, s2.length, "", ""]];
    while (stack.length && out.length < limit) {   // iterative DFS, same order as Python
      var t = stack.pop(), i = t[0], j = t[1];
      if (!i && !j) { out.push([t[2], t[3]]); continue; }
      var p = P[i][j];
      for (var k = p.length - 1; k >= 0; k--) {
        if (p[k] === D) stack.push([i - 1, j - 1, s1[i - 1] + t[2], s2[j - 1] + t[3]]);
        else if (p[k] === U) stack.push([i - 1, j, s1[i - 1] + t[2], "-" + t[3]]);
        else stack.push([i, j - 1, "-" + t[2], s2[j - 1] + t[3]]);
      }
    }
    return out;
  }

  function needlemanWunsch(s1, s2, sc) {
    var f = fill(s1, s2, sc, false), m = s1.length, n = s2.length;
    var t = traceback(s1, s2, f.F, f.P, [m, n], false);
    return { algorithm: GLOBAL, local: false, seq1: s1, seq2: s2, sc: sc, F: f.F, P: f.P, steps: f.steps,
      score: f.F[m][n], a1: t.a1, a2: t.a2, path: t.path, tb: t.steps, start: [0, 0], end: [m, n],
      optimalCount: countPaths(f.P, m, n), maxCells: [] };
  }

  function smithWaterman(s1, s2, sc) {
    var f = fill(s1, s2, sc, true), m = s1.length, n = s2.length, best = 0, end = [0, 0], maxCells = [];
    for (var i = 0; i <= m; i++) for (var j = 0; j <= n; j++) {
      if (f.F[i][j] > best) { best = f.F[i][j]; end = [i, j]; maxCells = [[i, j]]; }
      else if (f.F[i][j] === best && best > 0) maxCells.push([i, j]);
    }
    var base = { algorithm: LOCAL, local: true, seq1: s1, seq2: s2, sc: sc, F: f.F, P: f.P, steps: f.steps, optimalCount: null };
    if (best === 0) return Object.assign(base, { score: 0, a1: "", a2: "", path: [[0, 0]], start: [0, 0], end: [0, 0], maxCells: [],
      tb: [{ i: 0, j: 0, score: 0, move: null, options: "", p1: "", p2: "" }] });
    var t = traceback(s1, s2, f.F, f.P, end, true);
    return Object.assign(base, { score: best, a1: t.a1, a2: t.a2, path: t.path, tb: t.steps, start: t.path[0], end: end, maxCells: maxCells });
  }

  function stats(r) {
    var mt = 0, mm = 0, gp = 0, g1 = 0, g2 = 0, rescored = 0;
    for (var k = 0; k < r.a1.length; k++) {
      var a = r.a1[k], b = r.a2[k];
      if (a === "-" || b === "-") { gp++; if (a === "-") g1++; else g2++; rescored += r.sc.gap; }
      else if (a === b) { mt++; rescored += r.sc.match; } else { mm++; rescored += r.sc.mismatch; }
    }
    var len = r.a1.length;
    return { score: r.score, rescored: rescored, matches: mt, mismatches: mm, gaps: gp, gaps1: g1, gaps2: g2,
      length: len, identity: len ? Math.round(mt / len * 10000) / 100 : 0 };
  }

  function matchLine(a1, a2) {
    var s = "";
    for (var k = 0; k < a1.length; k++) s += (a1[k] === "-" || a2[k] === "-") ? " " : a1[k] === a2[k] ? "|" : ".";
    return s;
  }

  function randomDNA(n) { var s = ""; for (var k = 0; k < n; k++) s += "ACGT"[Math.floor(Math.random() * 4)]; return s; }

  var api = { GLOBAL: GLOBAL, LOCAL: LOCAL, clean: clean, validateSequence: validateSequence, validateScoring: validateScoring,
    needlemanWunsch: needlemanWunsch, smithWaterman: smithWaterman, enumerateGlobal: enumerateGlobal,
    stats: stats, matchLine: matchLine, randomDNA: randomDNA };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.DNA = api;
})(this);

// Zero-project-import Node.js replay of the Research A1 primary analysis.

import crypto from "node:crypto";
import fs from "node:fs";

const [portablePath, analysisPath, outputPath] = process.argv.slice(2);
if (!outputPath) throw new Error("usage: node research_a1_cleanroom.mjs PORTABLE ANALYSIS OUTPUT");

const MOD = 7;
const LAYERS = 5;
const FAMILIES = ["linear_target", "quadratic_target", "source_target"];
const SIZES = [5, 10, 15, 20, 25, 30, 40, 50];
const CRITICAL = 2.9551668474978343;
const NLL_SESOI = 0.01;
const NLL_MARGIN = 0.01;
const RECOVERY_SESOI = 0.10;
const RECOVERY_MARGIN = 0.05;

function mod(value) { return ((value % MOD) + MOD) % MOD; }
function close(left, right, tolerance = 1e-11) { return Math.abs(left - right) <= tolerance; }
function lexLess(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] < right[index];
  }
  return false;
}

function headValue(family, source, action, sourceIndex, target) {
  const active = action[sourceIndex];
  if (family === "linear_target") return mod(active * (1 + source[target]));
  if (family === "quadratic_target") return mod(active * (1 + source[target]) ** 2);
  if (family === "source_target") return mod(active * (1 + source[sourceIndex] + source[target]));
  throw new Error(`unknown family ${family}`);
}

function genericFeatures(source, action, graph) {
  const [target, sources] = graph;
  const values = [...action];
  for (const sourceIndex of sources) {
    for (const family of FAMILIES) values.push(headValue(family, source, action, sourceIndex, target));
  }
  return values.map(mod);
}

function increments(cases) {
  return cases.map((entry) => entry.observed.map((value, layer) => mod(value - entry.source[layer])));
}

function bestNonzero(rows) {
  let best = null;
  for (let coefficient = 1; coefficient < MOD; coefficient += 1) {
    let errors = 0;
    for (const [feature, observed] of rows) if (mod(coefficient * feature) !== observed) errors += 1;
    const candidate = [errors, coefficient];
    if (best === null || lexLess(candidate, best)) best = candidate;
  }
  return best;
}

function fitTyped(cases, graph) {
  const delta = increments(cases);
  const multipliers = [];
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const rows = [];
    for (let index = 0; index < cases.length; index += 1) {
      if (cases[index].action[layer] !== 0) rows.push([cases[index].action[layer], delta[index][layer]]);
    }
    multipliers.push(bestNonzero(rows)[1]);
  }
  const [target, sources] = graph;
  const families = [];
  const coefficients = [];
  for (const sourceIndex of sources) {
    let best = null;
    for (let familyIndex = 0; familyIndex < FAMILIES.length; familyIndex += 1) {
      const family = FAMILIES[familyIndex];
      const rows = [];
      for (let index = 0; index < cases.length; index += 1) {
        if (cases[index].action[sourceIndex] !== 0) {
          rows.push([headValue(family, cases[index].source, cases[index].action, sourceIndex, target), delta[index][target]]);
        }
      }
      const [errors, coefficient] = bestNonzero(rows);
      const candidate = [errors, familyIndex, coefficient];
      if (best === null || lexLess(candidate, best)) best = candidate;
    }
    families.push(FAMILIES[best[1]]);
    coefficients.push(best[2]);
  }
  return { kind: "typed", graph, families, multipliers, coefficients };
}

function fitIsomorphic(cases, graph) {
  const x = cases.map((entry) => genericFeatures(entry.source, entry.action, graph));
  const delta = increments(cases);
  const terms = [];
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const rows = [];
    for (let index = 0; index < cases.length; index += 1) {
      if (x[index][layer] !== 0) rows.push([x[index][layer], delta[index][layer]]);
    }
    terms.push([layer, layer, bestNonzero(rows)[1]]);
  }
  const [target, sources] = graph;
  for (let edge = 0; edge < sources.length; edge += 1) {
    let best = null;
    for (let familyIndex = 0; familyIndex < FAMILIES.length; familyIndex += 1) {
      const feature = LAYERS + edge * FAMILIES.length + familyIndex;
      const rows = [];
      for (let index = 0; index < cases.length; index += 1) {
        if (cases[index].action[sources[edge]] !== 0) rows.push([x[index][feature], delta[index][target]]);
      }
      const [errors, coefficient] = bestNonzero(rows);
      const candidate = [errors, familyIndex, coefficient];
      if (best === null || lexLess(candidate, best)) best = candidate;
    }
    terms.push([target, LAYERS + edge * FAMILIES.length + best[1], best[2]]);
  }
  return { kind: "generic", graph, terms };
}

function fitUnstructured(cases, graph) {
  const x = cases.map((entry) => genericFeatures(entry.source, entry.action, graph));
  const y = increments(cases);
  const prediction = cases.map(() => [0, 0, 0, 0, 0]);
  const available = new Set();
  for (let output = 0; output < LAYERS; output += 1) {
    for (let feature = 0; feature < 11; feature += 1) available.add(`${output}:${feature}`);
  }
  const terms = [];
  for (let step = 0; step < 7; step += 1) {
    const currentErrors = [];
    for (let output = 0; output < LAYERS; output += 1) {
      currentErrors.push(prediction.reduce((sum, row, index) => sum + Number(row[output] !== y[index][output]), 0));
    }
    let best = null;
    for (let output = 0; output < LAYERS; output += 1) {
      for (let feature = 0; feature < 11; feature += 1) {
        if (!available.has(`${output}:${feature}`)) continue;
        for (let coefficient = 1; coefficient < MOD; coefficient += 1) {
          let errors = 0;
          for (let index = 0; index < cases.length; index += 1) {
            if (mod(prediction[index][output] + coefficient * x[index][feature]) !== y[index][output]) errors += 1;
          }
          const candidate = [errors - currentErrors[output], output, feature, coefficient];
          if (best === null || lexLess(candidate, best)) best = candidate;
        }
      }
    }
    const [, output, feature, coefficient] = best;
    for (let index = 0; index < cases.length; index += 1) {
      prediction[index][output] = mod(prediction[index][output] + coefficient * x[index][feature]);
    }
    available.delete(`${output}:${feature}`);
    terms.push([output, feature, coefficient]);
  }
  return { kind: "generic", graph, terms };
}

function predict(model, source, action) {
  const delta = [0, 0, 0, 0, 0];
  if (model.kind === "typed") {
    const [target, sources] = model.graph;
    for (let layer = 0; layer < LAYERS; layer += 1) delta[layer] = model.multipliers[layer] * action[layer];
    for (let edge = 0; edge < 2; edge += 1) {
      delta[target] += model.coefficients[edge] * headValue(model.families[edge], source, action, sources[edge], target);
    }
  } else {
    const features = genericFeatures(source, action, model.graph);
    for (const [output, feature, coefficient] of model.terms) delta[output] += coefficient * features[feature];
  }
  return source.map((value, layer) => mod(value + delta[layer]));
}

function nll(model, cases) {
  const matched = Math.log(0.88);
  const mismatched = Math.log(0.02);
  let total = 0;
  for (const entry of cases) {
    const estimate = predict(model, entry.source, entry.action);
    for (let layer = 0; layer < LAYERS; layer += 1) total -= estimate[layer] === entry.observed[layer] ? matched : mismatched;
  }
  return total / (cases.length * LAYERS);
}

function trueTerms(spec) {
  const [target] = spec.graph;
  const terms = spec.multipliers.map((value, layer) => [layer, layer, value]);
  for (let edge = 0; edge < 2; edge += 1) {
    terms.push([target, LAYERS + edge * FAMILIES.length + FAMILIES.indexOf(spec.families[edge]), spec.coefficients[edge]]);
  }
  return terms;
}

function sameTerms(left, right) {
  const normalize = (rows) => rows.map((row) => row.join(":")).sort().join("|");
  return normalize(left) === normalize(right);
}

function summary(values) {
  const center = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1);
  const se = Math.sqrt(variance) / Math.sqrt(values.length);
  return { mean: center, lower: center - CRITICAL * se, upper: center + CRITICAL * se };
}

const portableBytes = fs.readFileSync(portablePath);
const portable = JSON.parse(portableBytes.toString("utf8"));
const published = JSON.parse(fs.readFileSync(analysisPath, "utf8"));
const portableHash = crypto.createHash("sha256").update(portableBytes).digest("hex");
const failures = [];
if (portableHash !== published.portable_replay_sha256) failures.push({ reason: "portable hash mismatch" });

const rows = [];
for (const world of portable.worlds) {
  const estimates = [];
  for (const size of SIZES) {
    const train = world.train.slice(0, size);
    const typed = fitTyped(train, world.spec.graph);
    const isomorphic = fitIsomorphic(train, world.spec.graph);
    const generic = fitUnstructured(train, world.spec.graph);
    const typedNll = nll(typed, world.test);
    const isomorphicNll = nll(isomorphic, world.test);
    const genericNll = nll(generic, world.test);
    const typedExact = JSON.stringify(typed.families) === JSON.stringify(world.spec.families)
      && JSON.stringify(typed.multipliers) === JSON.stringify(world.spec.multipliers)
      && JSON.stringify(typed.coefficients) === JSON.stringify(world.spec.coefficients);
    const genericExact = sameTerms(generic.terms, trueTerms(world.spec));
    if (!close(typedNll, isomorphicNll)) failures.push({ world: world.world_index, size, reason: "notation mismatch" });
    estimates.push({ size, nll: genericNll - typedNll, recovery: Number(typedExact) - Number(genericExact) });
  }
  rows.push(estimates);
}

const summaries = [];
for (let position = 0; position < SIZES.length; position += 1) {
  const nllSummary = summary(rows.map((row) => row[position].nll));
  const recoverySummary = summary(rows.map((row) => row[position].recovery));
  const advantage = nllSummary.lower > 0 && nllSummary.mean >= NLL_SESOI
    && recoverySummary.lower > 0 && recoverySummary.mean >= RECOVERY_SESOI;
  const equivalence = nllSummary.lower >= -NLL_MARGIN && nllSummary.upper <= NLL_MARGIN
    && recoverySummary.lower >= -RECOVERY_MARGIN && recoverySummary.upper <= RECOVERY_MARGIN;
  const expected = published.analysis.sample_size_summaries[position];
  const checks = [
    expected.sample_size === SIZES[position],
    close(nllSummary.mean, expected.generic_minus_typed_nll.mean),
    close(nllSummary.lower, expected.generic_minus_typed_nll.simultaneous_lower),
    close(nllSummary.upper, expected.generic_minus_typed_nll.simultaneous_upper),
    close(recoverySummary.mean, expected.typed_minus_generic_exact_recovery.mean),
    close(recoverySummary.lower, expected.typed_minus_generic_exact_recovery.simultaneous_lower),
    close(recoverySummary.upper, expected.typed_minus_generic_exact_recovery.simultaneous_upper),
    advantage === expected.joint_advantage,
    equivalence === expected.joint_equivalence,
  ];
  if (!checks.every(Boolean)) failures.push({ size: SIZES[position], reason: "published summary mismatch", checks });
  summaries.push({ sample_size: SIZES[position], nll: nllSummary, recovery: recoverySummary, advantage, equivalence });
}

const audit = {
  status: "cleanroom_nodejs_reimplementation",
  language_runtime: process.version,
  project_imports: 0,
  world_count: rows.length,
  primary_endpoint_count: 16,
  portable_sha256: portableHash,
  summaries,
  failures,
  passed: failures.length === 0,
  independent_replication: false,
};
fs.writeFileSync(outputPath, `${JSON.stringify(audit, null, 2)}\n`);
console.log(JSON.stringify({ status: audit.status, world_count: audit.world_count, failures: failures.length, passed: audit.passed }, null, 2));

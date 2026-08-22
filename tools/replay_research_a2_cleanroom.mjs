// Zero-project-import Node.js replay of all Research A2 endpoints and gates.

import crypto from "node:crypto";
import fs from "node:fs";

const [portablePath, rawPath, analysisPath, outputPath] = process.argv.slice(2);
if (!outputPath) {
  throw new Error(
    "usage: node replay_research_a2_cleanroom.mjs PORTABLE RAW ANALYSIS OUTPUT",
  );
}

const MOD = 7;
const LAYERS = 5;
const TYPED = ["linear_target", "quadratic_target", "source_target"];
const ALTERNATIVE = ["linear_target", "cubic_target", "source_target"];
const WIDTH_CRITICAL = 3.1969502291312533;
const NOISE_CRITICAL = 3.2790242977180215;
const SCOPE_CRITICAL = 2.638257273476751;
const NLL_SESOI = 0.01;
const RECOVERY_SESOI = 0.10;
const NLL_MARGIN = 0.01;
const RECOVERY_MARGIN = 0.05;
const SCOPE_NLL_SESOI = 0.10;
const SCOPE_ACCURACY_SESOI = 0.10;
const SCOPE_NLL_MARGIN = 0.01;
const SCOPE_ACCURACY_MARGIN = 0.025;
const TOLERANCE = 1e-12;

function mod(value) {
  return ((value % MOD) + MOD) % MOD;
}

function close(left, right) {
  return Math.abs(left - right) <= TOLERANCE;
}

function lexLess(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] < right[index];
  }
  return false;
}

function headValue(family, source, action, sourceIndex, target) {
  const active = action[sourceIndex];
  if (family === "linear_target") return mod(active * (1 + source[target]));
  if (family === "quadratic_target") {
    return mod(active * (1 + source[target]) ** 2);
  }
  if (family === "source_target") {
    return mod(active * (1 + source[sourceIndex] + source[target]));
  }
  if (family === "cubic_target") {
    return mod(active * (1 + source[target]) ** 3);
  }
  throw new Error(`unknown family ${family}`);
}

function catalogFeatures(source, action, graph, catalog) {
  const [target, sources] = graph;
  const values = [...action];
  for (const sourceIndex of sources) {
    for (const family of catalog) {
      values.push(headValue(family, source, action, sourceIndex, target));
    }
  }
  return values.map(mod);
}

const nuisanceDescriptors = [];
for (let state = 0; state < LAYERS; state += 1) {
  for (const degree of [1, 2]) {
    for (let action = 0; action < LAYERS; action += 1) {
      nuisanceDescriptors.push({ action, state, degree });
    }
  }
}

function widthFeatures(source, action, graph, positionCount) {
  const values = catalogFeatures(source, action, graph, TYPED);
  const extraCount = positionCount / LAYERS - values.length;
  for (const descriptor of nuisanceDescriptors.slice(0, extraCount)) {
    values.push(
      mod(action[descriptor.action] * source[descriptor.state] ** descriptor.degree),
    );
  }
  return values;
}

function increments(cases) {
  return cases.map((entry) =>
    entry.observed.map((value, layer) => mod(value - entry.source[layer])),
  );
}

function bestNonzero(rows) {
  let best = null;
  for (let coefficient = 1; coefficient < MOD; coefficient += 1) {
    let errors = 0;
    for (const [feature, observed] of rows) {
      if (mod(coefficient * feature) !== observed) errors += 1;
    }
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
      if (cases[index].action[layer] !== 0) {
        rows.push([cases[index].action[layer], delta[index][layer]]);
      }
    }
    multipliers.push(bestNonzero(rows)[1]);
  }
  const [target, sources] = graph;
  const families = [];
  const coefficients = [];
  for (const sourceIndex of sources) {
    let best = null;
    for (let familyIndex = 0; familyIndex < TYPED.length; familyIndex += 1) {
      const family = TYPED[familyIndex];
      const rows = [];
      for (let index = 0; index < cases.length; index += 1) {
        if (cases[index].action[sourceIndex] !== 0) {
          rows.push([
            headValue(
              family,
              cases[index].source,
              cases[index].action,
              sourceIndex,
              target,
            ),
            delta[index][target],
          ]);
        }
      }
      const [errors, coefficient] = bestNonzero(rows);
      const candidate = [errors, familyIndex, coefficient];
      if (best === null || lexLess(candidate, best)) best = candidate;
    }
    families.push(TYPED[best[1]]);
    coefficients.push(best[2]);
  }
  return { kind: "typed", graph, families, multipliers, coefficients };
}

function fitGeneric(cases, graph, featureSpec) {
  const featureRows = cases.map((entry) =>
    featureSpec.kind === "width"
      ? widthFeatures(entry.source, entry.action, graph, featureSpec.positionCount)
      : catalogFeatures(entry.source, entry.action, graph, featureSpec.catalog),
  );
  const targetRows = increments(cases);
  const prediction = cases.map(() => [0, 0, 0, 0, 0]);
  const featureCount = featureRows[0].length;
  const available = new Set();
  for (let output = 0; output < LAYERS; output += 1) {
    for (let feature = 0; feature < featureCount; feature += 1) {
      available.add(`${output}:${feature}`);
    }
  }
  const terms = [];
  for (let step = 0; step < 7; step += 1) {
    const currentErrors = [];
    for (let output = 0; output < LAYERS; output += 1) {
      currentErrors.push(
        prediction.reduce(
          (sum, row, index) =>
            sum + Number(row[output] !== targetRows[index][output]),
          0,
        ),
      );
    }
    let best = null;
    for (let output = 0; output < LAYERS; output += 1) {
      for (let feature = 0; feature < featureCount; feature += 1) {
        if (!available.has(`${output}:${feature}`)) continue;
        for (let coefficient = 1; coefficient < MOD; coefficient += 1) {
          let errors = 0;
          for (let index = 0; index < cases.length; index += 1) {
            const candidate = mod(
              prediction[index][output]
                + coefficient * featureRows[index][feature],
            );
            if (candidate !== targetRows[index][output]) errors += 1;
          }
          const candidate = [
            errors - currentErrors[output],
            output,
            feature,
            coefficient,
          ];
          if (best === null || lexLess(candidate, best)) best = candidate;
        }
      }
    }
    const [, output, feature, coefficient] = best;
    for (let index = 0; index < cases.length; index += 1) {
      prediction[index][output] = mod(
        prediction[index][output] + coefficient * featureRows[index][feature],
      );
    }
    available.delete(`${output}:${feature}`);
    terms.push([output, feature, coefficient]);
  }
  return { kind: "generic", graph, terms, featureSpec };
}

function predict(model, source, action) {
  const delta = [0, 0, 0, 0, 0];
  if (model.kind === "typed") {
    const [target, sources] = model.graph;
    for (let layer = 0; layer < LAYERS; layer += 1) {
      delta[layer] = model.multipliers[layer] * action[layer];
    }
    for (let edge = 0; edge < 2; edge += 1) {
      delta[target] +=
        model.coefficients[edge]
        * headValue(
          model.families[edge],
          source,
          action,
          sources[edge],
          target,
        );
    }
  } else {
    const features =
      model.featureSpec.kind === "width"
        ? widthFeatures(
            source,
            action,
            model.graph,
            model.featureSpec.positionCount,
          )
        : catalogFeatures(
            source,
            action,
            model.graph,
            model.featureSpec.catalog,
          );
    for (const [output, feature, coefficient] of model.terms) {
      delta[output] += coefficient * features[feature];
    }
  }
  return source.map((value, layer) => mod(value + delta[layer]));
}

function nll(model, cases) {
  const matched = Math.log(0.88);
  const mismatched = Math.log(0.02);
  let total = 0;
  for (const entry of cases) {
    const estimate = predict(model, entry.source, entry.action);
    for (let layer = 0; layer < LAYERS; layer += 1) {
      total -= estimate[layer] === entry.observed[layer] ? matched : mismatched;
    }
  }
  return total / (cases.length * LAYERS);
}

function centerAccuracy(model, cases) {
  let matches = 0;
  for (const entry of cases) {
    const estimate = predict(model, entry.source, entry.action);
    if (estimate.every((value, layer) => value === entry.center[layer])) matches += 1;
  }
  return matches / cases.length;
}

function sameArray(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function typedExact(model, spec) {
  return (
    sameArray(model.families, spec.families)
    && sameArray(model.multipliers, spec.multipliers)
    && sameArray(model.coefficients, spec.coefficients)
  );
}

function trueTerms(spec, catalog) {
  if (spec.families.some((family) => !catalog.includes(family))) return null;
  const [target] = spec.graph;
  const terms = spec.multipliers.map((value, layer) => [layer, layer, value]);
  for (let edge = 0; edge < 2; edge += 1) {
    terms.push([
      target,
      LAYERS + edge * catalog.length + catalog.indexOf(spec.families[edge]),
      spec.coefficients[edge],
    ]);
  }
  return terms;
}

function sameTerms(left, right) {
  if (right === null) return false;
  const normalize = (rows) => rows.map((row) => row.join(":")).sort().join("|");
  return normalize(left) === normalize(right);
}

function matchedRecord(spec, train, test, sampleSize, positionCount) {
  const prefix = train.slice(0, sampleSize);
  const typed = fitTyped(prefix, spec.graph);
  const generic = fitGeneric(prefix, spec.graph, {
    kind: "width",
    positionCount,
  });
  const typedRecovery = typedExact(typed, spec);
  const genericRecovery = sameTerms(generic.terms, trueTerms(spec, TYPED));
  const typedNll = nll(typed, test);
  const genericNll = nll(generic, test);
  return {
    world_index: spec.world_index,
    family_pair: spec.families,
    sample_size: sampleSize,
    position_count: positionCount,
    typed_exact: typedRecovery,
    generic_exact: genericRecovery,
    typed_nll: typedNll,
    generic_nll: genericNll,
    generic_minus_typed_nll: genericNll - typedNll,
    typed_minus_generic_exact:
      Number(typedRecovery) - Number(genericRecovery),
  };
}

function scopeRecord(spec, train, test, sampleSize) {
  const catalog = spec.condition === "matched" ? TYPED : ALTERNATIVE;
  const prefix = train.slice(0, sampleSize);
  const typed = fitTyped(prefix, spec.graph);
  const generic = fitGeneric(prefix, spec.graph, { kind: "catalog", catalog });
  const typedNll = nll(typed, test);
  const genericNll = nll(generic, test);
  return {
    world_index: spec.world_index,
    condition: spec.condition,
    family_pair: spec.families,
    sample_size: sampleSize,
    typed_exact: typedExact(typed, spec),
    generic_exact: sameTerms(generic.terms, trueTerms(spec, catalog)),
    generic_minus_typed_nll: genericNll - typedNll,
    typed_minus_generic_center_accuracy:
      centerAccuracy(typed, test) - centerAccuracy(generic, test),
  };
}

function summary(values, critical) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0)
    / (values.length - 1);
  const standardError = Math.sqrt(variance) / Math.sqrt(values.length);
  return {
    mean,
    lower: mean - critical * standardError,
    upper: mean + critical * standardError,
  };
}

function compareValue(actual, expected, path, failures) {
  if (typeof expected === "number") {
    if (!close(actual, expected)) failures.push({ path, actual, expected });
    return;
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) {
      failures.push({ path, reason: "array shape mismatch" });
      return;
    }
    expected.forEach((value, index) =>
      compareValue(actual[index], value, `${path}[${index}]`, failures),
    );
    return;
  }
  if (actual !== expected) failures.push({ path, actual, expected });
}

function compareRecords(actual, expected, axis, failures) {
  if (actual.length !== expected.length) {
    failures.push({ axis, reason: "record count mismatch", actual: actual.length, expected: expected.length });
    return;
  }
  for (let index = 0; index < expected.length; index += 1) {
    for (const [key, value] of Object.entries(expected[index])) {
      compareValue(actual[index][key], value, `${axis}[${index}].${key}`, failures);
    }
  }
}

function analyzeEfficiency(
  records,
  groupKey,
  groupValues,
  sizes,
  critical,
  requireEveryGroup,
  expected,
  failures,
) {
  const groupPassed = [];
  for (const groupValue of groupValues) {
    let anyAdvantage = false;
    for (const sampleSize of sizes) {
      const rows = records.filter(
        (row) =>
          row[groupKey] === groupValue && row.sample_size === sampleSize,
      );
      const nllSummary = summary(
        rows.map((row) => row.generic_minus_typed_nll),
        critical,
      );
      const recoverySummary = summary(
        rows.map((row) => row.typed_minus_generic_exact),
        critical,
      );
      const advantage =
        nllSummary.lower > 0
        && nllSummary.mean >= NLL_SESOI
        && recoverySummary.lower > 0
        && recoverySummary.mean >= RECOVERY_SESOI;
      const equivalence =
        nllSummary.lower >= -NLL_MARGIN
        && nllSummary.upper <= NLL_MARGIN
        && recoverySummary.lower >= -RECOVERY_MARGIN
        && recoverySummary.upper <= RECOVERY_MARGIN;
      anyAdvantage ||= advantage;
      const published = expected.summaries.find(
        (row) =>
          row[groupKey] === groupValue && row.sample_size === sampleSize,
      );
      if (!published) {
        failures.push({ groupKey, groupValue, sampleSize, reason: "published summary absent" });
        continue;
      }
      const checks = [
        [nllSummary.mean, published.generic_minus_typed_nll.mean, "nll.mean"],
        [nllSummary.lower, published.generic_minus_typed_nll.simultaneous_lower, "nll.lower"],
        [nllSummary.upper, published.generic_minus_typed_nll.simultaneous_upper, "nll.upper"],
        [recoverySummary.mean, published.typed_minus_generic_exact_recovery.mean, "recovery.mean"],
        [recoverySummary.lower, published.typed_minus_generic_exact_recovery.simultaneous_lower, "recovery.lower"],
        [recoverySummary.upper, published.typed_minus_generic_exact_recovery.simultaneous_upper, "recovery.upper"],
      ];
      for (const [actual, wanted, label] of checks) {
        if (!close(actual, wanted)) failures.push({ groupKey, groupValue, sampleSize, label, actual, expected: wanted });
      }
      if (advantage !== published.joint_advantage) failures.push({ groupKey, groupValue, sampleSize, reason: "advantage mismatch" });
      if (equivalence !== published.joint_equivalence) failures.push({ groupKey, groupValue, sampleSize, reason: "equivalence mismatch" });
    }
    groupPassed.push(anyAdvantage);
  }
  const gate = requireEveryGroup
    ? groupPassed.every(Boolean)
    : groupPassed.some(Boolean);
  if (gate !== expected.gate_passed) failures.push({ groupKey, reason: "axis gate mismatch", actual: gate, expected: expected.gate_passed });
  return gate;
}

function analyzeScope(records, sampleSize, expected, failures) {
  const summaries = {};
  for (const condition of ["matched", "typed_misspecified", "generic_misspecified"]) {
    const rows = records.filter(
      (row) => row.condition === condition && row.sample_size === sampleSize,
    );
    summaries[condition] = {
      nll: summary(rows.map((row) => row.generic_minus_typed_nll), SCOPE_CRITICAL),
      accuracy: summary(
        rows.map((row) => row.typed_minus_generic_center_accuracy),
        SCOPE_CRITICAL,
      ),
    };
    const published = expected.summaries[condition];
    for (const [name, local, remote] of [
      ["nll", summaries[condition].nll, published.generic_minus_typed_nll],
      ["accuracy", summaries[condition].accuracy, published.typed_minus_generic_center_accuracy],
    ]) {
      for (const [localKey, remoteKey] of [
        ["mean", "mean"],
        ["lower", "simultaneous_lower"],
        ["upper", "simultaneous_upper"],
      ]) {
        if (!close(local[localKey], remote[remoteKey])) failures.push({ condition, name, localKey, actual: local[localKey], expected: remote[remoteKey] });
      }
    }
  }
  const matched = summaries.matched;
  const typedMiss = summaries.typed_misspecified;
  const genericMiss = summaries.generic_misspecified;
  const gates = {
    matched_equivalence:
      matched.nll.lower >= -SCOPE_NLL_MARGIN
      && matched.nll.upper <= SCOPE_NLL_MARGIN
      && matched.accuracy.lower >= -SCOPE_ACCURACY_MARGIN
      && matched.accuracy.upper <= SCOPE_ACCURACY_MARGIN,
    typed_misspecified_favors_generic:
      typedMiss.nll.upper < 0
      && typedMiss.nll.mean <= -SCOPE_NLL_SESOI
      && typedMiss.accuracy.upper < 0
      && typedMiss.accuracy.mean <= -SCOPE_ACCURACY_SESOI,
    generic_misspecified_favors_typed:
      genericMiss.nll.lower > 0
      && genericMiss.nll.mean >= SCOPE_NLL_SESOI
      && genericMiss.accuracy.lower > 0
      && genericMiss.accuracy.mean >= SCOPE_ACCURACY_SESOI,
  };
  for (const [key, value] of Object.entries(gates)) {
    if (value !== expected.gates[key]) failures.push({ key, reason: "scope gate mismatch", actual: value, expected: expected.gates[key] });
  }
  return Object.values(gates).every(Boolean);
}

const portableBytes = fs.readFileSync(portablePath);
const portable = JSON.parse(portableBytes.toString("utf8"));
const raw = JSON.parse(fs.readFileSync(rawPath, "utf8"));
const published = JSON.parse(fs.readFileSync(analysisPath, "utf8"));
const portableHash = crypto
  .createHash("sha256")
  .update(portableBytes)
  .digest("hex");
const failures = [];
if (portableHash !== published.portable_replay_sha256) {
  failures.push({ reason: "portable hash mismatch", actual: portableHash, expected: published.portable_replay_sha256 });
}

const { design, axes } = portable;
const replay = { candidate_width: [], training_noise: [], misspecification: [] };
for (const world of axes.candidate_width) {
  for (const size of design.width_sample_sizes) {
    for (const width of design.width_position_counts) {
      replay.candidate_width.push(
        matchedRecord(world.spec, world.train, world.test, size, width),
      );
    }
  }
}
for (const world of axes.training_noise) {
  for (const probability of design.noise_probabilities) {
    const train = world.train_by_noise_probability[String(probability)];
    for (const size of design.noise_sample_sizes) {
      const record = matchedRecord(
        world.spec,
        train,
        world.test,
        size,
        design.width_position_counts[0],
      );
      record.train_noise_probability = probability;
      replay.training_noise.push(record);
    }
  }
}
for (const world of axes.misspecification) {
  replay.misspecification.push(
    scopeRecord(
      world.spec,
      world.train,
      world.test,
      design.scope_sample_size,
    ),
  );
}

compareRecords(replay.candidate_width, raw.axes.candidate_width, "candidate_width", failures);
compareRecords(replay.training_noise, raw.axes.training_noise, "training_noise", failures);
compareRecords(replay.misspecification, raw.axes.misspecification, "misspecification", failures);

const widthGate = analyzeEfficiency(
  replay.candidate_width,
  "position_count",
  design.width_position_counts,
  design.width_sample_sizes,
  WIDTH_CRITICAL,
  true,
  published.analysis.candidate_width,
  failures,
);
const noiseGate = analyzeEfficiency(
  replay.training_noise,
  "train_noise_probability",
  design.noise_probabilities,
  design.noise_sample_sizes,
  NOISE_CRITICAL,
  false,
  published.analysis.training_noise,
  failures,
);
const scopeGate = analyzeScope(
  replay.misspecification,
  design.scope_sample_size,
  published.analysis.misspecification_scope,
  failures,
);

const audit = {
  status: "cleanroom_nodejs_reimplementation",
  fixture_only: raw.status === "developmental_cleanroom_fixture",
  language_runtime: process.version,
  project_imports: 0,
  world_count_per_axis_or_scope_condition:
    portable.world_count_per_axis_or_scope_condition,
  endpoint_counts: { candidate_width: 36, training_noise: 48, scope: 6 },
  portable_sha256: portableHash,
  gates: { candidate_width: widthGate, training_noise: noiseGate, scope: scopeGate },
  failures,
  passed: failures.length === 0,
  independent_replication: false,
};
fs.writeFileSync(outputPath, `${JSON.stringify(audit, null, 2)}\n`);
console.log(
  JSON.stringify(
    {
      status: audit.status,
      fixture_only: audit.fixture_only,
      failures: failures.length,
      passed: audit.passed,
    },
    null,
    2,
  ),
);

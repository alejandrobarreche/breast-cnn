// Alcance (documentation/web-design.md, layer 4, blocks S1-S4): what the probability means,
// limitations, model card and glossary. Numbers come from manifest.json, results.json and
// runs.json.
import manifest from "./manifest.json";
import results from "./results.json";
import runs from "./runs.json";
import { mean, std, type Experiment } from "./lab";
import { make, spy, titular } from "./page";
import { confusion } from "./results";
import { summarize, type Patient } from "./split";
import { decimal, thousands } from "./svg";

const SECTIONS: [string, string][] = [["probabilidad", "Qué significa la probabilidad"], ["limitaciones", "Limitaciones"], ["ficha", "Ficha del modelo"], ["glosario", "Glosario"]];

const GLOSSARY: [string, string][] = [
  ["pCR (respuesta patológica completa)", "Ausencia de tumor invasivo residual en el tejido extirpado tras el tratamiento neoadyuvante. Es un indicador temprano de buena respuesta, no una garantía de curación."],
  ["Tratamiento neoadyuvante", "Quimioterapia u otro tratamiento sistémico que se administra antes de la cirugía, entre otras cosas para reducir el tumor."],
  ["DCE-MRI", "Resonancia magnética con contraste dinámico: una imagen antes de inyectar el contraste (PRE) y varias después (EARLY, LATE)."],
  ["Realce", "Aumento de la intensidad de un tejido al llegar el contraste. Aquí se mide como EARLY − PRE."],
  ["Curva de captación y lavado", "Evolución de la intensidad de un punto entre las fases. Hay lavado (washout) cuando la intensidad baja en la fase tardía."],
  ["Cohorte", "Grupo de pacientes de un mismo estudio (Duke, I-SPY1, I-SPY2), con su protocolo de adquisición y su proporción de pCR."],
  ["AUC", "Área bajo la curva ROC: probabilidad de que una paciente con pCR reciba una puntuación mayor que una sin pCR. 0,5 equivale al azar."],
  ["Fold y validación cruzada", "Las pacientes de entrenamiento se dividen en cinco grupos; cada uno valida una vez el modelo entrenado con los otros cuatro."],
  ["Calibración", "Transformación de la salida para que se pueda leer como probabilidad. Aquí, Platt: σ(a · logit + b) sobre la media de los cinco modelos."],
  ["Sensibilidad y especificidad", "Proporción de pacientes con pCR clasificadas como pCR, y proporción de pacientes sin pCR clasificadas como no pCR."],
  ["Mapa de oclusión", "Cambio de la probabilidad estimada al tapar cada región de la imagen. Indica qué mueve la salida del modelo, no dónde está el tumor."],
];

export function buildScope(root: HTMLElement, index: HTMLElement) {
  const s = manifest.test_patient, v = manifest.validation_slice, arch = manifest.architecture, t = manifest.training;
  const train = summarize(manifest.split.train as Patient[]);
  const finalRun = (runs.experiments as Experiment[]).find((e) => e.final)!;
  const test = results.test as number[][];
  const c = confusion(test.map((p) => p[0]), test.map((p) => p[2]), results.threshold);
  const sens = c.tp / (c.tp + c.fn), spec = c.tn / (c.tn + c.fp);
  const chance = s.auc.lower <= 0.5, indistinct = s.auc_difference.lower <= 0;
  const link = (href: string, label: string) => { const a = make("a", "", label); a.href = href; return a; };

  const head = make("header", "section-head");
  head.append(make("h1", "page-title", "Alcance"), make("p", "", "Lo que el valor devuelto permite afirmar, sus limitaciones y los términos necesarios para leerlo."));
  root.append(head);

  const evidence = make("p");
  evidence.append(
    `En test, por paciente, el AUC es ${decimal(s.auc.value)} (IC 95 % ${decimal(s.auc.lower)}–${decimal(s.auc.upper)})` +
      (chance && indistinct ? `, sin diferencia demostrada con el azar ni con una referencia que solo conoce la cohorte (${decimal(s.cohort_baseline_auc)}). ` : ". "),
    "El valor no permite afirmar que una paciente concreta alcanzará la pCR. Detalle en ",
    link("#estudio/resultados", "Resultados"),
    ".",
  );
  root.append(titular({
    id: "alcance-probabilidad",
    title: chance ? "Una estimación sin capacidad discriminativa demostrada" : "Una estimación con capacidad discriminativa limitada",
    body: [
      "El modelo devuelve la probabilidad de pCR que estima para un corte. Un valor de 0,30 significa P(pCR) = 0,30 según el modelo para ese corte: no es un 30 % de curación ni indica si la cirugía es necesaria.",
      evidence,
      "No es un diagnóstico ni orienta un tratamiento: la pCR solo se confirma con el estudio patológico tras la cirugía.",
    ],
  }));

  const limits = make("ul");
  limits.append(...[
    "Un único corte 2D por predicción, sin el volumen completo ni datos clínicos.",
    `Conjunto pequeño: ${thousands(train.patients)} pacientes de entrenamiento y ${s.patients} de test.`,
    "Las cohortes difieren en protocolo y en proporción de pCR; parte de lo que aprende la red puede ser la cohorte.",
    "El test se consultó dos veces, con el modelo anterior y con el actual: la segunda cifra no es una estimación del todo independiente.",
    "Sin validación externa: todas las pacientes proceden de BreastDCEDL.",
    `Con el umbral elegido, la mayoría de las pacientes con pCR se clasifican como no pCR (sensibilidad en test ${decimal(sens, 2)}).`,
    `Los ${manifest.examples.length} cortes de ejemplo de Analizar comprueban que la aplicación funciona, no su rendimiento.`,
  ].map((text) => make("li", "", text)));
  root.append(titular({ id: "alcance-limitaciones", title: "Limitaciones", body: [limits] }));

  const card: [string, string][] = [
    ["Uso previsto", "Docencia: ilustrar el entrenamiento y la evaluación de una CNN sobre DCE-MRI de mama."],
    ["Uso no previsto", "Cualquier decisión clínica."],
    ["Datos de entrenamiento", `BreastDCEDL: ${thousands(train.patients)} pacientes y ${thousands(train.slices)} cortes de Duke, I-SPY1 e I-SPY2`],
    ["Arquitectura", `CNN 2D de ${arch.stages.length} etapas, canales ${arch.stages[0].channels} → ${arch.stages[arch.stages.length - 1].channels}, recorte central ${arch.input_crop} × ${arch.input_crop}`],
    ["Parámetros", `${thousands(arch.params)} por modelo, ${arch.models} modelos (uno por fold)`],
    ["Entrenamiento", `${t.experiment}: ${t.loss}, ${t.optimizer === "adamw" ? "AdamW" : t.optimizer}, ${t.epochs} épocas`],
    ["Entrada", "float32 [3, 256, 256]: PRE, EARLY y LATE en escala value / 255"],
    ["Salida", "Probabilidad de pCR calibrada (Platt) sobre la media de los modelos"],
    ["Umbral", `${decimal(manifest.model.threshold)} por corte y ${decimal(results.threshold)} por paciente (especificidad ≥ ${decimal(manifest.model.min_specificity, 1)} en validación)`],
    ["Validación interna, por corte", `AUC ${decimal(v.auc)}, sensibilidad ${decimal(v.sensitivity, 2)}, especificidad ${decimal(v.specificity, 2)}`],
    ["Validación interna, por paciente", `AUC ${decimal(mean(finalRun.auc))} ± ${decimal(std(finalRun.auc))} entre folds`],
    ["Test, por paciente", `AUC ${decimal(s.auc.value)} (IC 95 % ${decimal(s.auc.lower)}–${decimal(s.auc.upper)}), sensibilidad ${decimal(sens, 2)}, especificidad ${decimal(spec, 2)}`],
    ["Formato", `ONNX, ${decimal(manifest.model.bytes / 1e6, 1)} MB, ejecutado con ONNX Runtime Web en WebAssembly`],
    ["SHA-256", manifest.model.sha256],
  ];
  const table = make("table", "table");
  const body = make("tbody");
  for (const [k, value] of card) {
    const tr = make("tr");
    const th = make("td", "", k);
    th.style.color = "var(--ink-2)";
    th.style.width = "34%";
    const td = make("td", "", value);
    if (k === "SHA-256") td.style.wordBreak = "break-all";
    tr.append(th, td);
    body.append(tr);
  }
  table.append(body);
  root.append(titular({ id: "alcance-ficha", title: "Ficha del modelo", body: [table] }));

  const glossary = make("dl", "glossary");
  for (const [term, definition] of GLOSSARY) {
    const item = make("div");
    item.append(make("dt", "", term), make("dd", "", definition));
    glossary.append(item);
  }
  root.append(titular({ id: "alcance-glosario", title: "Glosario", body: [glossary] }));

  const links = SECTIONS.map(([id, label], k) => {
    const a = make("a");
    a.href = `#alcance/${id}`;
    a.append(make("span", "", String(k + 1)), label);
    index.append(a);
    return a;
  });
  spy(links, SECTIONS.map(([id]) => document.getElementById(`alcance-${id}`)!));
}

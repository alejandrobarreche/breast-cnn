// Layer table of the network as built by breast_cnn.models.builder: per stage, `convs`
// convolutions k × k without bias, each followed by BatchNorm and ReLU, then 2 × 2 max pooling;
// then global average pooling, dropout and a linear layer to one logit. The theoretical
// receptive field grows by (k − 1)·jump per convolution and by jump per pooling, where jump is
// the input distance between neighbouring units (doubled by each pooling).

export interface StageSpec { channels: number; convs: number; size: number; params: number }

export interface Layer { name: string; shape: string; params: number }

export interface Step {
  title: string;
  layers: Layer[];
  params: number;
  /** Theoretical receptive field in input pixels and the input distance between output units. */
  field: number;
  jump: number;
  outSize: number;
}

const times = (...v: number[]) => v.join(" × ");

export function steps(stages: StageSpec[], kernel: number, inChannels: number, crop: number, models: number): Step[] {
  const out: Step[] = [{
    title: "Entrada y recorte",
    layers: [
      { name: "Tensor de entrada (PRE, EARLY, LATE)", shape: times(inChannels, 256, 256), params: 0 },
      { name: `Recorte central ${crop} × ${crop}`, shape: times(inChannels, crop, crop), params: 0 },
    ],
    params: 0, field: 1, jump: 1, outSize: crop,
  }];
  let channels = inChannels, field = 1, jump = 1;
  stages.forEach((stage, i) => {
    const layers: Layer[] = [];
    for (let c = 0; c < stage.convs; c++) {
      layers.push({ name: `Convolución ${kernel} × ${kernel}, ${channels} → ${stage.channels}`, shape: times(stage.channels, stage.size, stage.size), params: kernel * kernel * channels * stage.channels });
      layers.push({ name: "BatchNorm y ReLU", shape: times(stage.channels, stage.size, stage.size), params: 2 * stage.channels });
      channels = stage.channels;
      field += (kernel - 1) * jump;
    }
    layers.push({ name: "Max pooling 2 × 2", shape: times(stage.channels, stage.size / 2, stage.size / 2), params: 0 });
    field += jump;
    jump *= 2;
    out.push({ title: `Etapa ${i + 1}`, layers, params: layers.reduce((a, l) => a + l.params, 0), field, jump, outSize: stage.size / 2 });
  });
  out.push({
    title: "Decisión",
    layers: [
      { name: "Promedio global por canal", shape: String(channels), params: 0 },
      { name: "Dropout", shape: String(channels), params: 0 },
      { name: `Lineal ${channels} → 1`, shape: "1 logit", params: channels + 1 },
    ],
    params: channels + 1, field, jump, outSize: 1,
  });
  out.push({
    title: "Calibración y media",
    layers: [
      { name: `Platt en cada uno de los ${models} modelos: σ(a · logit + b)`, shape: `${models} probabilidades`, params: 0 },
      { name: `Media de las ${models} probabilidades calibradas`, shape: "P(pCR)", params: 0 },
    ],
    params: 0, field, jump, outSize: 1,
  });
  return out;
}

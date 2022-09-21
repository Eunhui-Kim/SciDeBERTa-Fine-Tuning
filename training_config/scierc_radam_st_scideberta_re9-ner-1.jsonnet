local template = import "template.deberta.libsonnet";

template.DyGIE {
  deberta_model: "/scratch/hpc64a01/workspace/pretrained-model/scideberta-abs",
  cuda_device: 0,
  data_paths: {
    train: "data/scierc/normalized_data/json/train.json",
    validation: "data/scierc/normalized_data/json/dev.json",
    test: "data/scierc/normalized_data/json/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation: 0.0,
    coref: 1.0,
    events: 0.0
  },
  model +: {
    modules +: {
      coref +: {
        coref_prop: 1
      }
    },
  },
  trainer +: {
     optimizer +: {
         type: 'radam'
	  },
  },
  target_task: "ner",
}

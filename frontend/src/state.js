export const state = {
  colabUrl: '',
  connected: false,
  pipeline: 'ohrc_nac',
  pairsByPipeline: {},  // pipeline id -> [pair ids] from /api/pipelines
  lastRun: null,        // { pipeline, pair_id } of the run whose results are shown
  running: false,
  runError: null,       // failure reported by the server during the current run
  currentStep: 0,
  stepStates: {},       // step id -> 'pending' | 'running' | 'done' | 'error'
  stepDetails: {},      // step id -> { detail, image }
  results: null,        // metrics of the finished run
  resultImages: {},     // result tab -> image data URL
  flowCols: 0,          // boxes per row in the current flow layout
  openStep: null,       // step whose popup is open
  healthCheck: null,    // interval id of the connection check during a run
};

public abstract class ResidualPredictor<Model>
{
    public List<Sample> samples;
    public float learningRate;   
    public List<float[]> accumulators;
    public List<Sample> residuals;
    public List<Model> models;

    public ResidualPredictor(List<Sample> samples, float learningRate, int modelCount)
    {
        this.samples = samples;
        this.learningRate = learningRate;
        this.accumulators = new List<float[]>(samples.Count);
        this.residuals = new List<Sample>(samples.Count);
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            Sample sample = samples[sampleIndex];
            accumulators.Add(new float[sample.output.Length]);
            float[] residual = new float[sample.output.Length];
            for (int i = 0; i < sample.output.Length; i++)
            {
                residual[i] = sample.output[i] - accumulators[sampleIndex][i];
            }
            residuals.Add(new Sample(sample.input, residual));
        }
        this.models = new List<Model>();
        for (int modelIndex = 0; modelIndex < modelCount; modelIndex++)
        {
            AddModel();
        }
    }

    protected abstract (Model model, List<float[]> predictions) AddModel(List<Sample> residuals);
    
    public void AddModel()
    {
        (Model model, List<float[]> predictions) = AddModel(residuals);
        models.Add(model);
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            Sample sample = samples[sampleIndex];
            Sample residualSample = residuals[sampleIndex];
            float[] accumulator = accumulators[sampleIndex];
            float[] prediction = predictions[sampleIndex];
            for (int j = 0; j < prediction.Length; j++)
            {
                accumulator[j] += learningRate * prediction[j];
                residualSample.output[j] = sample.output[j] - accumulator[j];
            }
        }
    }

    protected abstract float[] Predict(Model model, float[] input);

    public float[] Predict(float[] input)
    {
        float[] accumulator = new float[samples[0].output.Length];
        foreach (Model model in models)
        {
            float[] prediction = Predict(model, input);
            for (int i = 0; i < prediction.Length; i++)
            {
                accumulator[i] += learningRate * prediction[i];
            }
        }
        return accumulator;
    }
}
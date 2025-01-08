public class ResidualAggregator : Model
{
    public List<Model> models;
    public float learningRate;
    public int? outputLength;
    public bool verbose;

    public ResidualAggregator(float learningRate, List<Model> models, bool verbose = false)
    {
        this.models = models;
        this.learningRate = learningRate;
        this.outputLength = null;
        this.verbose = verbose;
    }

    public override void Fit(List<Sample> samples, List<int> features)
    {
        // mark the output length (used in Predict)
        this.outputLength = samples[0].output.Length;

        // create accumulators and residuals
        List<float[]> accumulators = new List<float[]>(samples.Count);
        List<Sample> residuals = new List<Sample>(samples.Count);

        // establish baseline residuals from zeroed accumulators
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

        // iterate through models
        for (int modelIndex = 0; modelIndex < models.Count; modelIndex++)
        {
            if (verbose)
            {
                Console.WriteLine($"ResidualAggregator Fitting model {modelIndex + 1} of {models.Count}");
            }

            // get model
            Model model = models[modelIndex];

            // fit model to residuals
            model.Fit(residuals, features);

            // if this isnt the last model
            if (modelIndex < models.Count - 1)
            {
                // predict residuals
                List<float[]> residualPredictions = new List<float[]>(samples.Count);
                foreach (Sample residual in residuals)
                {
                    residualPredictions.Add(model.Predict(residual.input));
                }

                // update accumulators and residuals
                for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
                {
                    Sample sample = samples[sampleIndex];
                    Sample residualSample = residuals[sampleIndex];
                    float[] accumulator = accumulators[sampleIndex];
                    float[] residualPrediction = residualPredictions[sampleIndex];
                    for (int j = 0; j < residualPrediction.Length; j++)
                    {
                        accumulator[j] += learningRate * residualPrediction[j];
                        residualSample.output[j] = sample.output[j] - accumulator[j];
                    }
                }
            }
        }
    }

    public override float[] Predict(float[] input)
    {
        if (outputLength == null)
        {
            throw new Exception("Model has not been fit");
        }
        float[] accumulator = new float[outputLength.Value];
        foreach (Model model in models)
        {
            float[] residualPrediction = model.Predict(input);
            for (int j = 0; j < residualPrediction.Length; j++)
            {
                accumulator[j] += learningRate * residualPrediction[j];
            }
        }
        return accumulator;
    }
}
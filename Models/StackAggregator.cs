public class StackAggregator : Model
{
    public List<Model> models;

    public StackAggregator(List<Model> models)
    {
        this.models = models;
    }

    public override void Fit(List<Sample> samples, List<int> features)
    {
        // create a list of next samples
        List<Sample> nextSamples = new List<Sample>(samples);

        // iterate through models in the stack
        for (int modelIndex = 0; modelIndex < models.Count; modelIndex++)
        {
            // get model at index
            Model model = models[modelIndex];

            // fit model to next samples
            model.Fit(nextSamples, features);

            // if this isnt the last model in the stack
            if (modelIndex < models.Count - 1)
            {
                // predict the next samples
                List<float[]> predictions = new List<float[]>(nextSamples.Count);
                foreach (Sample sample in nextSamples)
                {
                    predictions.Add(model.Predict(sample.input));
                }

                // replace the next samples inputs with the predictions
                for (int sampleIndex = 0; sampleIndex < nextSamples.Count; sampleIndex++)
                {
                    Sample sample = nextSamples[sampleIndex];
                    float[] prediction = predictions[sampleIndex];
                    nextSamples[sampleIndex] = new Sample(prediction, sample.output);
                }
            }
        }
    }

    public override float[] Predict(float[] input)
    {
        // create a prediction which feeds through the stack
        float[] prediction = input;

        // iterate through models in the stack
        for (int modelIndex = 0; modelIndex < models.Count; modelIndex++)
        {
            // get model at index
            Model model = models[modelIndex];

            // predict the next input
            prediction = model.Predict(prediction);
        }

        // return the prediction
        return prediction;
    }
}
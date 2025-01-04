public static class Error
{
    public delegate float ErrorFunction(List<Sample> samples, List<float[]> predictions);

    public static float MeanSquaredError(List<Sample> samples, List<float[]> predictions)
    {
        float error = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            float[] actual = samples[i].output;
            float[] prediction = predictions[i];
            for (int j = 0; j < actual.Length; j++)
            {
                error += MathF.Pow(actual[j] - prediction[j], 2);
            }
        }
        return error / (float)samples.Count;
    }

    public static float MeanAbsoluteError(List<Sample> samples, List<float[]> predictions)
    {
        float error = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            float[] actual = samples[i].output;
            float[] prediction = predictions[i];
            for (int j = 0; j < actual.Length; j++)
            {
                error += MathF.Abs(actual[j] - prediction[j]);
            }
        }
        return error / (float)samples.Count;
    }

    public static float ArgmaxError(List<Sample> samples, List<float[]> predictions)
    {
        int errors = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            float[] actual = samples[i].output;
            float[] prediction = predictions[i];
            int actualIndex = -1;
            float actualValue = float.NegativeInfinity;
            int predictionIndex = -1;
            float predictionValue = float.NegativeInfinity;
            for (int j = 0; j < actual.Length; j++)
            {
                if (actual[j] > actualValue)
                {
                    actualValue = actual[j];
                    actualIndex = j;
                }
                if (prediction[j] > predictionValue)
                {
                    predictionValue = prediction[j];
                    predictionIndex = j;
                }
            }
            if (actualIndex != predictionIndex)
            {
                errors++;
            }
        }
        return (float)errors / (float)samples.Count;
    }
}

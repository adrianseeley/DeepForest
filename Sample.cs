public class Sample
{
    public float[] input;
    public float[] output;

    public Sample(float[] input, float[] output)
    {
        this.input = input;
        this.output = output;
    }

    public static List<Sample> Normalize(List<Sample> samples)
    {
        float[] inputMins = new float[samples[0].input.Length];
        float[] inputMaxs = new float[samples[0].input.Length];
        float[] outputMins = new float[samples[0].output.Length];
        float[] outputMaxs = new float[samples[0].output.Length];
        for (int i = 0; i < samples[0].input.Length; i++)
        {
            inputMins[i] = float.PositiveInfinity;
            inputMaxs[i] = float.NegativeInfinity;
        }
        for (int i = 0; i < samples[0].output.Length; i++)
        {
            outputMins[i] = float.PositiveInfinity;
            outputMaxs[i] = float.NegativeInfinity;
        }
        foreach (Sample sample in samples)
        {
            for (int i = 0; i < sample.input.Length; i++)
            {
                if (sample.input[i] < inputMins[i])
                {
                    inputMins[i] = sample.input[i];
                }
                if (sample.input[i] > inputMaxs[i])
                {
                    inputMaxs[i] = sample.input[i];
                }
            }
            for (int i = 0; i < sample.output.Length; i++)
            {
                if (sample.output[i] < outputMins[i])
                {
                    outputMins[i] = sample.output[i];
                }
                if (sample.output[i] > outputMaxs[i])
                {
                    outputMaxs[i] = sample.output[i];
                }
            }
        }
        float[] inputRanges = new float[samples[0].input.Length];
        float[] outputRanges = new float[samples[0].output.Length];
        for (int i = 0; i < samples[0].input.Length; i++)
        {
            inputRanges[i] = inputMaxs[i] - inputMins[i];
        }
        for (int i = 0; i < samples[0].output.Length; i++)
        {
            outputRanges[i] = outputMaxs[i] - outputMins[i];
        }

        List<Sample> normalized = new List<Sample>();
        foreach (Sample sample in samples)
        {
            float[] input = new float[sample.input.Length];
            for (int i = 0; i < sample.input.Length; i++)
            {
                input[i] = (sample.input[i] - inputMins[i]) / inputRanges[i];
            }
            float[] output = new float[sample.output.Length];
            for (int i = 0; i < sample.output.Length; i++)
            {
                output[i] = (sample.output[i] - outputMins[i]) / outputRanges[i];
            }
            normalized.Add(new Sample(input, output));
        }
        return normalized;
    }
}

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
                if (inputRanges[i] == 0f)
                {
                    input[i] = 0f;
                }
                else
                {
                    input[i] = (sample.input[i] - inputMins[i]) / inputRanges[i];
                }
            }
            float[] output = new float[sample.output.Length];
            for (int i = 0; i < sample.output.Length; i++)
            {
                if (outputRanges[i] == 0f)
                {
                    output[i] = 0f;
                }
                else
                {
                    output[i] = (sample.output[i] - outputMins[i]) / outputRanges[i];
                }
            }
            normalized.Add(new Sample(input, output));
        }
        return normalized;
    }

    public static (List<Sample> normalizedSamplesA, List<Sample> normalizedSamplesB) Conormalize(List<Sample> samplesA, List<Sample> samplesB)
    {
        List<Sample> jointSamples = new List<Sample>();
        jointSamples.AddRange(samplesA);
        jointSamples.AddRange(samplesB);
        List<Sample> normalizedJointSamples = Normalize(jointSamples);
        List<Sample> normalizedSamplesA = normalizedJointSamples.GetRange(0, samplesA.Count);
        List<Sample> normalizedSamplesB = normalizedJointSamples.GetRange(samplesA.Count, samplesB.Count);
        return (normalizedSamplesA, normalizedSamplesB);
    }
}

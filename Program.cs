using System.Net.Http.Headers;
using System.Reflection;

public class Program
{
    public static List<Sample> ReadMNIST (string filename, int max = -1)
    {
        List<Sample> samples = new List<Sample>();
        string[] lines = File.ReadAllLines(filename);
        for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++) // skip headers
        {
            string line = lines[lineIndex].Trim();
            if (line.Length == 0)
            {
                continue; // skip empty lines
            }
            string[] parts = line.Split(',');
            int labelInt = int.Parse(parts[0]);
            float[] labelOneHot = new float[10];
            labelOneHot[labelInt] = 1;
            float[] input = new float[parts.Length - 1];
            for (int i = 1; i < parts.Length; i++)
            {
                input[i - 1] = float.Parse(parts[i]);
            }
            samples.Add(new Sample(input, labelOneHot));
            if (max != -1 && samples.Count >= max)
            {
                break;
            }
        }
        return samples;
    }

    public static float GetError(List<Sample> normalizedTrain, List<Sample> normalizedTest, int k, float[] weights, int outputLength)
    {
        float errorSum = 0;
        object errorLock = new object();
        Parallel.For(0, normalizedTest.Count, testIndex =>
        {
            Sample testSample = normalizedTest[testIndex];
            List<(Sample trainSample, float distance)> trainSampleDistances = new List<(Sample, float)>(normalizedTrain.Count);
            for (int trainIndex = 0; trainIndex < normalizedTrain.Count; trainIndex++)
            {
                float distance = Utility.WeightedMeanAbsoluteDistance(testSample.input, normalizedTrain[trainIndex].input, weights);
                trainSampleDistances.Add((normalizedTrain[trainIndex], distance));
            }
            trainSampleDistances.Sort((a, b) => a.distance.CompareTo(b.distance));

            float[] output = new float[outputLength];
            float weightSum = 0f;
            for (int trainIndex = 0; trainIndex < k; trainIndex++)
            {
                Sample trainSample = trainSampleDistances[trainIndex].trainSample;
                float distance = trainSampleDistances[trainIndex].distance;
                float weight = 1f / (distance + 0.0000001f);
                weightSum += weight;
                for (int i = 0; i < outputLength; i++)
                {
                    output[i] += trainSample.output[i] * weight;
                }
            }
            for (int i = 0; i < outputLength; i++)
            {
                output[i] /= weightSum;
            }
            float error = Error.MeanAbsoluteError(testSample.output, output);
            lock(errorLock)
            {
                errorSum += error;
            }
        });
        errorSum /= normalizedTest.Count;
        return errorSum;
    }

    public static void Main()
    {
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        (List<Sample> normalizedTrain, List<Sample> normalizedTest) = Sample.Conormalize(mnistTrain, mnistTest);

        TextWriter tw = new StreamWriter("./results.csv");
        tw.WriteLine("epoch,weightSum,error");

        int inputLength = normalizedTrain[0].input.Length;
        int outputLength = normalizedTrain[0].output.Length;
        int k = 3;
        float decrement = 0.01f;
        float[] weights = new float[inputLength];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = 1f;
        }
        int epoch = 0;
        float bestError = GetError(normalizedTrain, normalizedTest, k, weights, outputLength);
        tw.WriteLine(epoch + "," + weights.Sum() + "," + bestError);
        tw.Flush();

        // keep making passes until no improvements
        bool improved = true;
        while(improved)
        {
            // reset improved flag
            improved = false;

            // walk through input weights to decrement them
            for (int inputIndex = 0; inputIndex < inputLength; inputIndex++)
            {
                // if this input is already at or below 0, we cant decrement it
                if (weights[inputIndex] <= 0f)
                {
                    continue;
                }

                // while we can decrement further
                while (weights[inputIndex] > 0f)
                {
                    Console.WriteLine("Trying to decrement input " + inputIndex + " from " + weights[inputIndex]);
                    epoch++;

                    // decrement
                    weights[inputIndex] -= decrement;
                    if (weights[inputIndex] < 0f)
                    {
                        weights[inputIndex] = 0f;
                    }

                    // score it
                    float error = GetError(normalizedTrain, normalizedTest, k, weights, outputLength);

                    // record it
                    tw.WriteLine(epoch + "," + weights.Sum() + "," + error);
                    tw.Flush();

                    // if this is worse
                    if (error > bestError)
                    {
                        Console.WriteLine("Worse: " + error);

                        // undo the decrement
                        weights[inputIndex] += decrement;
                        
                        // break trying to decrement this index
                        break;
                    }
                    // otherwise it was the same or better
                    else
                    {
                        Console.WriteLine("Same or Better: " + error);

                        // update score
                        bestError = error;

                        // mark that we improved
                        improved = true;

                        // continue decrementing it
                        continue;
                    }
                }

            }
        }

        Console.WriteLine("Most correct: " + bestError);
        Console.WriteLine("Weights:");
        for (int i = 0; i < weights.Length; i++)
        {
            Console.WriteLine("Input " + i + ": " + weights[i]);
        }


        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}
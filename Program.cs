using System.Net.Http.Headers;
using System.Reflection;
using static System.Runtime.InteropServices.JavaScript.JSType;

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

    public static (float error, int correct) GetError(List<Sample> normalizedTrain, List<Sample> normalizedTest, int k, float[] trainSampleDistanceWeights, float[] inputWeights, int outputLength)
    {
        int correct = 0;
        float[] errors = new float[normalizedTest.Count];
        Parallel.For(0, normalizedTest.Count, testIndex =>
        {
            Sample testSample = normalizedTest[testIndex];
            List<(Sample trainSample, float distance)> trainSampleDistances = new List<(Sample, float)>(normalizedTrain.Count);
            for (int trainIndex = 0; trainIndex < normalizedTrain.Count; trainIndex++)
            {
                float distanceWeight = trainSampleDistanceWeights[trainIndex];
                float distance = Utility.WeightedMeanAbsoluteDistance(testSample.input, normalizedTrain[trainIndex].input, inputWeights) * distanceWeight;
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
            errors[testIndex] = error;
            if (Error.ArgmaxEquals(testSample, output))
            {
                Interlocked.Increment(ref correct);
            }
        });
        float averageError = (float)errors.Sum() / (float)normalizedTest.Count;
        return (averageError, correct);
    }

    public static void Main()
    {
        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        (List<Sample> normalizedTrain, List<Sample> normalizedTest) = Sample.Conormalize(mnistTrain, mnistTest);

        TextWriter tw = new StreamWriter("./results.csv");
        tw.WriteLine("epoch,trainSampleDistanceWeightSum,inputWeightSum,error,correct");

        int inputLength = normalizedTrain[0].input.Length;
        int outputLength = normalizedTrain[0].output.Length;
        int k = 3;
        float nudge = 0.1f;
        float[] trainSampleDistanceWeights = new float[normalizedTrain.Count];
        for (int i = 0; i < normalizedTrain.Count; i++)
        {
            trainSampleDistanceWeights[i] = 1f;
        }
        float trainSampleDistanceWeightSum = trainSampleDistanceWeights.Sum();
        float[] inputWeights = new float[inputLength];
        for (int i = 0; i < inputWeights.Length; i++)
        {
            inputWeights[i] = 1f;
        }
        float inputWeightSum = inputWeights.Sum();
        int epoch = 0;
        (float bestError, int bestCorrect) = GetError(normalizedTrain, normalizedTest, k, trainSampleDistanceWeights, inputWeights, outputLength);
        tw.WriteLine($"{epoch},{trainSampleDistanceWeightSum},{inputWeightSum},{bestError},{bestCorrect}");
        tw.Flush();

        // try to reduce input weights
        bool improved = true;
        while(improved)
        {
            // reset improved flag
            improved = false;

            // walk through input inputWeights to nudge them
            for (int inputIndex = 0; inputIndex < inputLength; inputIndex++)
            {
                // if this input is already at or below 0, we cant nudge it
                if (inputWeights[inputIndex] <= 0f)
                {
                    continue;
                }

                // while we can nudge further
                while (inputWeights[inputIndex] > 0f)
                {
                    Console.WriteLine("Trying Decrement InputWeight: " + inputIndex + " From: " + inputWeights[inputIndex]);
                    epoch++;

                    // decrement
                    inputWeights[inputIndex] -= nudge;
                    if (inputWeights[inputIndex] < 0f)
                    {
                        inputWeights[inputIndex] = 0f;
                    }

                    // get weight sum
                    inputWeightSum = inputWeights.Sum();

                    // score it
                    (float error, int correct) = GetError(normalizedTrain, normalizedTest, k, trainSampleDistanceWeights, inputWeights, outputLength);

                    // record it
                    tw.WriteLine($"{epoch},{trainSampleDistanceWeightSum},{inputWeightSum},{error},{correct}");
                    tw.Flush();

                    // if this is worse
                    if (error > bestError)
                    {
                        Console.WriteLine($"Worse Error: {error} Correct: {correct} TrainWeightSum: {trainSampleDistanceWeightSum} InputWeightSum: {inputWeightSum}");

                        // increment back into place
                        inputWeights[inputIndex] += nudge;
                        
                        // break trying to nudge this index
                        break;
                    }
                    // otherwise it was the same or better
                    else
                    {
                        Console.WriteLine($"Better Error: {error} Correct: {correct} TrainWeightSum: {trainSampleDistanceWeightSum} InputWeightSum: {inputWeightSum}");

                        // update score
                        bestError = error;
                        bestCorrect = correct;

                        // mark that we improved
                        improved = true;

                        // continue decrementing it
                        continue;
                    }
                }

            }
        }

        // try to reduce train sample distance weights
        trainSampleDistanceWeightSum = trainSampleDistanceWeights.Sum();
        inputWeightSum = inputWeights.Sum();
        improved = true;
        while (improved)
        {
            // reset improved flag
            improved = false;

            // walk through train sample distance weights to nudge them
            for (int trainIndex = 0; trainIndex < normalizedTrain.Count; trainIndex++)
            {
                // if this weight is already 0 cant nudge
                if (trainSampleDistanceWeights[trainIndex] <= 0f)
                {
                    continue;
                }

                // while we can nudge further
                while (trainSampleDistanceWeights[trainIndex] > 0f)
                {
                    Console.WriteLine("Trying Decrement Train Index: " + trainIndex + " From: " + trainSampleDistanceWeights[trainIndex]);
                    epoch++;

                    // increment
                    trainSampleDistanceWeights[trainIndex] += nudge;

                    // get weight sum
                    trainSampleDistanceWeightSum = trainSampleDistanceWeights.Sum();

                    // score it
                    (float error, int correct) = GetError(normalizedTrain, normalizedTest, k, trainSampleDistanceWeights, inputWeights, outputLength);

                    // record it
                    tw.WriteLine($"{epoch},{trainSampleDistanceWeightSum},{inputWeightSum},{error},{correct}");
                    tw.Flush();

                    // if this is worse
                    if (error > bestError)
                    {
                        Console.WriteLine($"Worse Error: {error} Correct: {correct} TrainWeightSum: {trainSampleDistanceWeightSum} InputWeightSum: {inputWeightSum}");

                        // decrement back into place
                        inputWeights[trainIndex] -= nudge;

                        // break trying to nudge this index
                        break;
                    }
                    // otherwise it was the same or better
                    else
                    {
                        Console.WriteLine($"Better Error: {error} Correct: {correct} TrainWeightSum: {trainSampleDistanceWeightSum} InputWeightSum: {inputWeightSum}");

                        // update score
                        bestError = error;
                        bestCorrect = correct;

                        // mark that we improved
                        improved = true;

                        // continue decrementing it
                        continue;
                    }
                }
            }
        }

        trainSampleDistanceWeightSum = trainSampleDistanceWeights.Sum();
        inputWeightSum = inputWeights.Sum();
        Console.WriteLine($"Final Error: {bestError} Correct: {bestCorrect} TrainWeightSum: {trainSampleDistanceWeightSum} InputWeightSum: {inputWeightSum}");

        TextWriter weightsWriter = new StreamWriter("./weights.csv");
        weightsWriter.WriteLine("inputIndex,weight");
        for (int i = 0; i < inputWeights.Length; i++)
        {
            weightsWriter.WriteLine($"{i},{inputWeights[i]}");
        }
        weightsWriter.WriteLine("trainIndex,weight");
        for (int i = 0; i < trainSampleDistanceWeights.Length; i++)
        {
            weightsWriter.WriteLine($"{i},{trainSampleDistanceWeights[i]}");
        }

        Console.WriteLine("Press return to exit");
        Console.ReadLine();
    }
}
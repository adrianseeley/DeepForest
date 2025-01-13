public class Program
{
    public static List<Sample> ReadMNIST(string filename, int max = -1)
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

    public static float[] CreateInputWeights(List<Sample> trainSamples)
    {
        float[] inputWeights = new float[trainSamples[0].input.Length];
        for (int i = 0; i < inputWeights.Length; i++)
        {
            inputWeights[i] = 1f;
        }
        return inputWeights;
    }

    public static float Distance(float[] a, float[] b, float[] w)
    {
        float distance = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            distance += MathF.Pow(a[i] - b[i], 2) * w[i];
        }
        return distance;
    }

    public static float ReweightDistance(float distance, float a, float b, float oldW, float newW)
    {
        distance -= MathF.Pow(a - b, 2) * oldW;
        distance += MathF.Pow(a - b, 2) * newW;
        return distance;
    }

    public static float[,] CreateTrainTrainDistanceMatrix(List<Sample> trainSamples, float[] inputWeights)
    {
        float[,] distanceMatrix = new float[trainSamples.Count, trainSamples.Count];
        int complete = 0;
        Parallel.For(0, trainSamples.Count, a =>
        {
            float[] inputA = trainSamples[a].input;
            distanceMatrix[a, a] = 0f;
            for (int b = a + 1; b < trainSamples.Count; b++)
            {
                float[] inputB = trainSamples[b].input;
                float distance = Distance(inputA, inputB, inputWeights);   
                distanceMatrix[a, b] = distance;
                distanceMatrix[b, a] = distance;
            }
            int localComplete = Interlocked.Increment(ref complete);
            if (localComplete % 1000 == 0)
            {
                Console.WriteLine($"TrainTrainMatrix: {localComplete} / {trainSamples.Count}");
            }
        });
        return distanceMatrix;
    }

    public static float[,] CreateTrainTestDistanceMatrix(List<Sample> trainSamples, List<Sample> testSamples, float[] inputWeights)
    {
        float[,] distanceMatrix = new float[trainSamples.Count, testSamples.Count];
        int complete = 0;
        Parallel.For(0, trainSamples.Count, trainIndex =>
        {
            float[] inputTrain = trainSamples[trainIndex].input;
            for (int testIndex = 0; testIndex < testSamples.Count; testIndex++)
            {
                float[] inputTest = testSamples[testIndex].input;
                float distance = Distance(inputTrain, inputTest, inputWeights);
                distanceMatrix[trainIndex, testIndex] = distance;
            }
            int localComplete = Interlocked.Increment(ref complete);
            if (localComplete % 1000 == 0)
            {
                Console.WriteLine($"TrainTestMatrix: {localComplete} / {trainSamples.Count}");
            }
        });
        return distanceMatrix;
    }

    public static void ModifyTrainTrainInputWeight(List<Sample> trainSamples, float[] inputWeights, float[,] trainTrainDistanceMatrix, int inputWeightIndex, float oldInputWeightValue, float newInputWeightValue)
    {
        Parallel.For(0, trainSamples.Count, a =>
        {
            float[] inputA = trainSamples[a].input;
            for (int b = a + 1; b < trainSamples.Count; b++)
            {
                float[] inputB = trainSamples[b].input;
                float distance = ReweightDistance(trainTrainDistanceMatrix[a, b], inputA[inputWeightIndex], inputB[inputWeightIndex], oldInputWeightValue, newInputWeightValue);
                trainTrainDistanceMatrix[a, b] = distance;
                trainTrainDistanceMatrix[b, a] = distance;
            }
        });
        inputWeights[inputWeightIndex] = newInputWeightValue;
    }

    public static void ModifyTrainTestInputWeight(List<Sample> trainSamples, List<Sample> testSamples, float[] inputWeights, float[,] trainTestDistanceMatrix, int inputWeightIndex, float oldInputWeightValue, float newInputWeightValue)
    {
        Parallel.For(0, trainSamples.Count, trainIndex =>
        {
            float[] inputTrain = trainSamples[trainIndex].input;
            for (int testIndex = 0; testIndex < testSamples.Count; testIndex++)
            {
                float[] inputTest = testSamples[testIndex].input;
                float distance = ReweightDistance(trainTestDistanceMatrix[trainIndex, testIndex], inputTrain[inputWeightIndex], inputTest[inputWeightIndex], oldInputWeightValue, newInputWeightValue);
                trainTestDistanceMatrix[trainIndex, testIndex] = distance;
            }
        });
        inputWeights[inputWeightIndex] = newInputWeightValue;
    }

    public static List<(Sample trainSample, float distance)> FindNeighboursTrain(int k, int trainIndex, List<Sample> trainSamples, float[,] trainTrainDistanceMatrix)
    {
        List<(Sample trainSample, float distance)> neighbours = new List<(Sample, float)>(k);
        for (int neighbourTrainIndex = 0; neighbourTrainIndex < trainSamples.Count; neighbourTrainIndex++)
        {
            if (neighbourTrainIndex == trainIndex)
            {
                continue; // cant test against self
            }
            float distance = trainTrainDistanceMatrix[neighbourTrainIndex, trainIndex];
            if (neighbours.Count < k)
            {
                neighbours.Add((trainSamples[neighbourTrainIndex], distance));
                neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
                continue;
            }
            if (distance < neighbours.Last().distance)
            {
                neighbours.Add((trainSamples[neighbourTrainIndex], distance));
                neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
                neighbours.RemoveAt(neighbours.Count - 1);
                continue;
            }
        }
        return neighbours;
    }

    public static List<(Sample trainSample, float distance)> FindNeighboursTest(int k, int testIndex, List<Sample> trainSamples, float[,] trainTestDistanceMatrix)
    {
        List<(Sample trainSample, float distance)> neighbours = new List<(Sample, float)>(k);
        for (int trainIndex = 0; trainIndex < trainSamples.Count; trainIndex++)
        {
            float distance = trainTestDistanceMatrix[trainIndex, testIndex];
            if (neighbours.Count < k)
            {
                neighbours.Add((trainSamples[trainIndex], distance));
                neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
                continue;
            }
            if (distance < neighbours.Last().distance)
            {
                neighbours.Add((trainSamples[trainIndex], distance));
                neighbours.Sort((a, b) => a.distance.CompareTo(b.distance));
                neighbours.RemoveAt(neighbours.Count - 1);
                continue;
            }
        }
        return neighbours;
    }

    public static float[] Predict(int k, List<(Sample trainSample, float distance)> neighbours)
    {
        int predictionSize = neighbours[0].trainSample.output.Length;
        float weightSum = 0f;
        float[] prediction = new float[predictionSize];
        for (int neighbourIndex = 0; neighbourIndex < neighbours.Count; neighbourIndex++)
        {
            Sample neighbour = neighbours[neighbourIndex].trainSample;
            float distance = neighbours[neighbourIndex].distance;
            float weight = 1f / (distance + 0.0000001f);
            weightSum += weight;
            for (int j = 0; j < predictionSize; j++)
            {
                prediction[j] += neighbour.output[j] * weight;
            }
        }
        for (int j = 0; j < predictionSize; j++)
        {
            prediction[j] /= weightSum;
        }
        return prediction;
    }

    public static (float distanceSum, int correct) EvaluateTrain(int k, List<Sample> trainSamples, List<int> trainSamplesArgmax, float[,] trainTrainDistanceMatrix)
    {
        float[] localDistanceSums = new float[trainSamples.Count];
        int correct = 0;
        Parallel.For(0, trainSamples.Count, trainIndex =>
        {
            float[] output = trainSamples[trainIndex].output;
            List<(Sample trainSample, float distance)> neighbours = FindNeighboursTrain(k, trainIndex, trainSamples, trainTrainDistanceMatrix);
            float[] prediction = Predict(k, neighbours);
            float localDistanceSum = 0f;
            for (int j = 0; j < prediction.Length; j++)
            {
                localDistanceSum += MathF.Abs(prediction[j] - output[j]);
            }
            localDistanceSums[trainIndex] = localDistanceSum;
            if (Argmax(prediction) == trainSamplesArgmax[trainIndex])
            {
                Interlocked.Increment(ref correct);
            }
        });
        float distanceSum = 0f;
        for (int i = 0; i < trainSamples.Count; i++)
        {
            distanceSum += localDistanceSums[i];
        }
        return (distanceSum, correct);
    }

    public static (float distanceSum, int correct) EvaluateTest(int k, List<Sample> trainSamples, List<Sample> testSamples, List<int> testSamplesArgmax, float[,] trainTestDistanceMatrix)
    {
        float[] localDistanceSums = new float[testSamples.Count];
        int correct = 0;
        Parallel.For(0, testSamples.Count, testIndex =>
        {
            float[] output = testSamples[testIndex].output;
            List<(Sample trainSample, float distance)> neighbours = FindNeighboursTest(k, testIndex, trainSamples, trainTestDistanceMatrix);
            float[] prediction = Predict(k, neighbours);
            float localDistanceSum = 0f;
            for (int j = 0; j < prediction.Length; j++)
            {
                localDistanceSum += MathF.Abs(prediction[j] - output[j]);
            }
            localDistanceSums[testIndex] = localDistanceSum;
            if (Argmax(prediction) == testSamplesArgmax[testIndex])
            {
                Interlocked.Increment(ref correct);
            }
        });
        float distanceSum = 0f;
        for (int i = 0; i < testSamples.Count; i++)
        {
            distanceSum += localDistanceSums[i];
        }
        return (distanceSum, correct);
    }

    public static int Argmax(float[] vector)
    {
        int index = 0;
        float value = vector[0];
        for (int i = 1; i < vector.Length; i++)
        {
            if (vector[i] > value)
            {
                index = i;
                value = vector[i];
            }
        }
        return index;
    }

    public static void Log(TextWriter csv, int epoch, int fails, float trainDistanceSum, int trainCorrect, float testDistanceSum, int testCorrect, float[] inputWeights)
    {
        float inputWeightSum = inputWeights.Sum();
        csv.WriteLine($"{epoch},{fails},{trainDistanceSum},{trainCorrect},{testDistanceSum},{testCorrect},{inputWeightSum}");
        csv.Flush();
        Console.WriteLine($"epoch: {epoch}, fails: {fails}, trainDistanceSum: {trainDistanceSum}, trainCorrect: {trainCorrect}, testDistanceSum: {testDistanceSum}, testCorrect: {testCorrect}, inputWeightSum: {inputWeightSum}");
    }

    public static void Main()
    {
        Random random = new Random();
        float nudgeAmount = 0.01f;
        int k = 3;
        int epoch = 0;
        int fails = 0;

        List<Sample> mnistTrain = ReadMNIST("D:/data/mnist_train.csv", max: 1000);
        List<Sample> mnistTest = ReadMNIST("D:/data/mnist_test.csv", max: 1000);
        (List<Sample> trainSamples, List<Sample> testSamples) = Sample.Conormalize(mnistTrain, mnistTest);

        List<int> trainSamplesArgmax = new List<int>(trainSamples.Count);
        foreach (Sample sample in trainSamples)
        {
            trainSamplesArgmax.Add(Argmax(sample.output));
        }
        List<int> testSamplesArgmax = new List<int>(testSamples.Count);
        foreach (Sample sample in testSamples)
        {
            testSamplesArgmax.Add(Argmax(sample.output));
        }

        float[] inputWeights = CreateInputWeights(trainSamples);
        float[,] trainTrainDistanceMatrix = CreateTrainTrainDistanceMatrix(trainSamples, inputWeights);
        float[,] trainTestDistanceMatrix = CreateTrainTestDistanceMatrix(trainSamples, testSamples, inputWeights);
        List<int> mutationIndices = Enumerable.Range(0, inputWeights.Length).ToList();

        TextWriter csv = new StreamWriter("results.csv", append: false);
        csv.WriteLine("epoch,fails,trainDistanceSum,trainCorrect,testDistanceSum,testCorrect,inputWeightSum");
        csv.Flush();

        // calculate starting point
        (float trainDistanceSum, int trainCorrect) = EvaluateTrain(k, trainSamples, trainSamplesArgmax, trainTrainDistanceMatrix);
        (float testDistanceSum, int testCorrect) = EvaluateTest(k, trainSamples, testSamples, testSamplesArgmax, trainTestDistanceMatrix);

        // log starting point
        Log(csv, epoch, fails, trainDistanceSum, trainCorrect, testDistanceSum, testCorrect, inputWeights);

        // loop until improvements stop
        bool improved = true;
        while (improved)
        {
            improved = false;

            // shuffle mutations
            for (int i = 0; i < mutationIndices.Count; i++)
            {
                int j = random.Next(i, mutationIndices.Count);
                (mutationIndices[i], mutationIndices[j]) = (mutationIndices[j], mutationIndices[i]); 
            }

            // iterate shuffled mutations
            foreach (int mutationIndex in mutationIndices)
            {
                // get the old inputWeight
                float oldWeight = inputWeights[mutationIndex];

                // if its already 0 we cant reduce it more
                if (oldWeight == 0f)
                {
                    continue;
                }

                // determine the new inputWeight
                float newWeight = oldWeight - nudgeAmount;
                if (newWeight < 0f)
                {
                    newWeight = 0f;
                }

                // modify the inputWeight
                ModifyTrainTrainInputWeight(trainSamples, inputWeights, trainTrainDistanceMatrix, mutationIndex, oldWeight, newWeight);
                ModifyTrainTestInputWeight(trainSamples, testSamples, inputWeights, trainTestDistanceMatrix, mutationIndex, oldWeight, newWeight);

                // evaluate the mutation on train
                (float mutationTrainDistanceSum, int mutationTrainCorrect) = EvaluateTrain(k, trainSamples, trainSamplesArgmax, trainTrainDistanceMatrix);

                // if we are keeping the mutationIndex
                if (mutationTrainDistanceSum <= trainDistanceSum)
                {
                    // increment epoch
                    epoch++;

                    // score and mark it
                    improved = true;
                    trainDistanceSum = mutationTrainDistanceSum;
                    trainCorrect = mutationTrainCorrect;

                    // evaluate the mutation on test
                    (testDistanceSum, testCorrect) = EvaluateTest(k, trainSamples, testSamples, testSamplesArgmax, trainTestDistanceMatrix);

                    // log it
                    Log(csv, epoch, fails, trainDistanceSum, trainCorrect, testDistanceSum, testCorrect, inputWeights);
                    continue;
                }

                // otherwise we need to undo the mutationIndex
                ModifyTrainTrainInputWeight(trainSamples, inputWeights, trainTrainDistanceMatrix, mutationIndex, newWeight, oldWeight);
                ModifyTrainTestInputWeight(trainSamples, testSamples, inputWeights, trainTestDistanceMatrix, mutationIndex, newWeight, oldWeight);

                // increment fails
                fails++;
            }
        }

    }
}
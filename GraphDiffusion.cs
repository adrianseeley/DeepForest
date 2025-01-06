/// <summary>
/// Samples are assumed to be prenormalized in both the input and output dimensions.
/// Diffusion rate must be a value between 0 and 1 exclusive (it cannot be 0 or 1) otherwise no diffusion will occur.
/// </summary>

public class GraphDiffusion
{
    public List<GraphDiffusionNode> nodes;

    public GraphDiffusion(List<Sample> samples, float diffusionRate, int diffusionSteps)
    {
        // calculate contra diffusion rate
        float contraDiffusionRate = 1f - diffusionRate;

        // convert all samples to nodes
        nodes = new List<GraphDiffusionNode>();
        foreach (Sample sample in samples)
        {
            nodes.Add(new GraphDiffusionNode(sample));
        }

        // debug draw
        DebugDrawSamples();
        DebugDrawNonDiffused();

        // we can add non-producing nodes here if needed
        Random random = new Random();
        for (int i = 0; i < 3000; i++)
        {
            float rx = random.NextSingle();
            float ry = random.NextSingle();
            nodes.Add(new GraphDiffusionNode(new float[] { rx, ry }, nodes[0].output.Length));
        }

        // calculate distances between all nodes (we skip a==b which defaults to 0 anyways)
        float[,] distances = new float[nodes.Count, nodes.Count];
        float distanceMin = float.PositiveInfinity;
        float distanceMax = float.NegativeInfinity;
        for (int a = 0; a < nodes.Count; a++)
        {
            for (int b = a + 1; b < nodes.Count; b++)
            {
                float distance = EuclideanDistance(nodes[a].input, nodes[b].input);
                distances[a, b] = distance;
                distances[b, a] = distance;
                distanceMin = MathF.Min(distanceMin, distance);
                distanceMax = MathF.Max(distanceMax, distance);
            }
        }
        float distanceRange = distanceMax - distanceMin;

        // calculate normalized distances between all nodes (again skipping a==b which defaults to 0)
        float[,] normalizedDistances = new float[nodes.Count, nodes.Count];
        for (int a = 0; a < nodes.Count; a++)
        {
            for (int b = a + 1; b < nodes.Count; b++)
            {
                float normalizedDistance = (distances[a, b] - distanceMin) / distanceRange;
                normalizedDistances[a, b] = normalizedDistance;
                normalizedDistances[b, a] = normalizedDistance;
            }
        }

        // create targets
        List<float[]> nodeTargets = new List<float[]>();
        foreach (GraphDiffusionNode node in nodes)
        {
            nodeTargets.Add(new float[node.output.Length]);
        }

        // make diffusion steps
        for (int diffusionStep = 0; diffusionStep < diffusionSteps; diffusionStep++)
        {
            // debug draw
            DebugDraw(diffusionStep);

            // iterate through nodes to generate the targets
            for (int nodeIndex = 0; nodeIndex < nodes.Count; nodeIndex++)
            {
                // get the node target
                float[] nodeTarget = nodeTargets[nodeIndex];

                // clear it
                Array.Clear(nodeTarget, 0, nodeTarget.Length);

                // track the influence weight sum
                float influenceWeightSum = 0f;

                // iterate through all other nodes to influence the target
                for (int otherNodeIndex = 0; otherNodeIndex < nodes.Count; otherNodeIndex++)
                {
                    // get the other node
                    GraphDiffusionNode otherNode = nodes[otherNodeIndex];

                    // for now influence is 1 - normalized distance (this will be updated later)
                    float influenceWeight = 1f - normalizedDistances[nodeIndex, otherNodeIndex];

                    if (influenceWeight < 0.9f)
                    {
                        influenceWeight *= 0.01f;
                    }

                    // if this other node has a sample, its a producer
                    if (otherNode.sample != null)
                    {
                        // add the sample output influence to the target
                        for (int i = 0; i < nodeTarget.Length; i++)
                        {
                            nodeTarget[i] += otherNode.sample.output[i] * influenceWeight;
                        }

                        // add the influence weight to the sum
                        influenceWeightSum += influenceWeight;
                    }

                    // all nodes are transmitters, add their output to the target
                    for (int i = 0; i < nodeTarget.Length; i++)
                    {
                        nodeTarget[i] += otherNode.output[i] * influenceWeight;
                    }

                    // add the influence weight to the sum
                    influenceWeightSum += influenceWeight;
                }

                // weighted average the target by the influence weight sum
                for (int i = 0; i < nodeTarget.Length; i++)
                {
                    nodeTarget[i] /= influenceWeightSum;
                }
            }

            // iterate through nodes to update the outputs based on targets
            for (int nodeIndex = 0; nodeIndex < nodes.Count; nodeIndex++)
            {
                // get the node
                GraphDiffusionNode node = nodes[nodeIndex];

                // get the node target
                float[] nodeTarget = nodeTargets[nodeIndex];

                // update the node output
                for (int i = 0; i < node.output.Length; i++)
                {
                    float existingContribution = node.output[i] * contraDiffusionRate;
                    float targetContribution = nodeTarget[i] * diffusionRate;
                    node.output[i] = existingContribution + targetContribution;
                }
            }
        }
    }

    public static int debugWidth = 500;
    public static int debugHeight = 500;
    public static int debugK = 1;

    public void DebugDrawSamples()
    {
        int width = debugWidth;
        int height = debugHeight;
        byte[,,] bitmap = Bitmap.Create(width, height);

        // draw samples
        foreach (GraphDiffusionNode node in nodes)
        {
            if (node.sample == null)
            {
                continue;
            }
            int x = (int)(node.input[0] * width);
            int y = (int)(node.input[1] * height);
            byte r = (byte)(node.sample.output[0] * 255);
            byte b = (byte)(node.sample.output[1] * 255);
            Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
        }

        Bitmap.SaveToBMP($"./bmp/samples.bmp", bitmap);
    }

    public void DebugDrawNonDiffused()
    {
        int k = debugK;
        int width = debugWidth;
        int height = debugHeight;
        byte[,,] bitmap = Bitmap.Create(width, height);

        // draw predictions
        for (int x = 0; x < width; x++)
        {
            Console.Write($"\rfixed x: {x}/{width}");
            Parallel.For(0, height, y =>
            {
                float nx = (float)x / (float)width;
                float ny = (float)y / (float)height;
                float[] prediction = LookupKAverageNonDiffused(new float[] { nx, ny }, k);
                byte r = (byte)(prediction[0] * 255);
                byte b = (byte)(prediction[1] * 255);
                Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
            });
        }

        // draw samples
        foreach (GraphDiffusionNode node in nodes)
        {
            if (node.sample == null)
            {
                continue;
            }
            int x = (int)(node.input[0] * width);
            int y = (int)(node.input[1] * height);
            byte r = (byte)(node.sample.output[0] * 255);
            byte b = (byte)(node.sample.output[1] * 255);
            Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
        }

        Bitmap.SaveToBMP($"./bmp/nondiffused.bmp", bitmap);
        Console.WriteLine();
    }

    public void DebugDraw(int diffusionStep)
    {
        int k = debugK;
        int width = debugWidth;
        int height = debugHeight;
        byte[,,] bitmap = Bitmap.Create(width, height);
        
        // draw predictions
        for (int x = 0; x < width; x++)
        {
            Console.Write($"\rds: {diffusionStep} x: {x}/{width}");
            Parallel.For(0, height, y =>
            {
                float nx = (float)x / (float)width;
                float ny = (float)y / (float)height;
                float[] prediction = LookupKAverage(new float[] { nx, ny }, k);
                byte r = (byte)(prediction[0] * 255);
                byte b = (byte)(prediction[1] * 255);
                Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
            });
        }

        // draw samples
        foreach(GraphDiffusionNode node in nodes)
        {
            if (node.sample == null)
            {
                continue;
            }
            int x = (int)(node.input[0] * width);
            int y = (int)(node.input[1] * height);
            byte r = (byte)(node.sample.output[0] * 255);
            byte b = (byte)(node.sample.output[1] * 255);
            Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
        }

        Bitmap.SaveToBMP($"./bmp/diffusion_{diffusionStep}.bmp", bitmap);
        Console.WriteLine();
    }

    public float[] LookupKAverageNonDiffused(float[] input, int k)
    {
        List<(GraphDiffusionNode node, float distance)> nodeDistances = new List<(GraphDiffusionNode node, float distance)>();
        foreach (GraphDiffusionNode node in nodes)
        {
            if (node.sample == null)
            {
                continue;
            }
            float distance = EuclideanDistance(input, node.input);
            nodeDistances.Add((node, distance));
        }
        nodeDistances.Sort((a, b) => a.distance.CompareTo(b.distance));
        int minK = Math.Min(k, nodeDistances.Count);
        float[] output = new float[nodes[0].output.Length];
        for (int i = 0; i < minK; i++)
        {
            GraphDiffusionNode node = nodeDistances[i].node;
            if (node.sample == null)
            {
                throw new Exception("Unreachable.");
            }
            for (int j = 0; j < output.Length; j++)
            {
                output[j] += node.sample.output[j];
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= minK;
        }
        return output;
    }

    public float[] LookupKAverage(float[] input, int k)
    {
        List<(GraphDiffusionNode node, float distance)> nodeDistances = new List<(GraphDiffusionNode node, float distance)>();
        foreach (GraphDiffusionNode node in nodes)
        {
            float distance = EuclideanDistance(input, node.input);
            nodeDistances.Add((node, distance));
        }
        nodeDistances.Sort((a, b) => a.distance.CompareTo(b.distance));
        int minK = Math.Min(k, nodeDistances.Count);
        float[] output = new float[nodes[0].output.Length];
        for (int i = 0; i < minK; i++)
        {
            GraphDiffusionNode node = nodeDistances[i].node;
            for (int j = 0; j < output.Length; j++)
            {
                output[j] += node.output[j];
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= minK;
        }
        return output;
    }

    public static float EuclideanDistance(float[] a, float[] b)
    {
        float distance = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            distance += MathF.Pow(a[i] - b[i], 2);
        }
        return MathF.Sqrt(distance);
    }
}

public class GraphDiffusionNode
{
    public float[] input;
    public float[] output;
    public Sample? sample;

    public GraphDiffusionNode(float[] input, int outputSize)
    {
        this.input = input;
        this.output = new float[outputSize];
        this.sample = null;
    }

    public GraphDiffusionNode(Sample sample)
    {
        this.input = new float[sample.input.Length];
        sample.input.CopyTo(this.input, 0);
        this.output = new float[sample.output.Length];
        this.sample = sample;
    }
}
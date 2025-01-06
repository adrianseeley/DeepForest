full set
ResidualRandomForest rrf = new ResidualRandomForest(train, treeCount: 10000, minSamplesPerLeaf: (int)MathF.Sqrt(train.Count), splitAttempts: 100, learningRate: 0.001f, threadCount: 12, verbose: true);
Error: 0.0476 
Error: 0.0453 with all the dupes

dropout, only worse outcomes
duplication is out
flip is out
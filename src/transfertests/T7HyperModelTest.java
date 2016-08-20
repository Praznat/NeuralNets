package transfertests;

import java.util.Collection;

import ann.BiasNode;
import ann.FFNeuralNetwork;
import ann.Node;
import ann.indirectencodings.RelationManager;
import ann.testing.FlierCatcher;
import modeler.ModelLearner;
import modeler.ModelLearnerHeavy;
import modeler.ModelLearnerModularImpure;
import modeler.ModelLearnerModularPure;
import modeler.TransitionMemory;
import modulemanagement.ModuleDisplayer;
import modulemanagement.ModuleManagerImpure;
import modulemanagement.ModuleManagerPure;
import modulemanagement.ReusableNNModule;

@SuppressWarnings("unused")
public class T7HyperModelTest {
	static int size = 5;
	static int trainTurns = 1000;
	static int learnIterations = 100;
	
	public static void main(String[] args) {
//		testModules();
		sanityCheck();
//		testTransferGridSize(new StringBuilder(), 100, 1);
//		testABunch();
		
//		StringBuilder result = new StringBuilder();
//		testCatcherOnlyToFliers(result, 1000, 1);
//		System.out.println(result);
	}
	
	private static void testABunch() {
		StringBuilder result = new StringBuilder();
		for (int ss : new int[] {25, 50, 75, 100, 200, 500, 1000}) {
			result.append("sample size:	" + ss + "\n");
			for (int i = 0; i < 10; i++) testTransferGridSize(result, ss, 8);
		}
		System.out.println(result);
	}
	
	private static void sanityCheck() {
//		FlierCatcher game = new FlierCatcher(size);
////		game.setSpawnFreq(0);
////		game.modeler = T6BasicWeightSharing.createModelerWithWgtSharing(game, size*2, trainTurns, true);
//		T5GeoNet.makeModelerGeoNet(game, trainTurns, new int[] {3});
//		System.out.println("Training source...");
//		game.modeler.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
//		Node output0 = game.modeler.getModelVTA().getNeuralNetwork().getOutputNodes().get(0);
//		Node output5 = game.modeler.getModelVTA().getNeuralNetwork().getOutputNodes().get(5);
//		Node output10 = game.modeler.getModelVTA().getNeuralNetwork().getOutputNodes().get(10);
//		RelationManager<Node> relMngr = RelationManager.createFromGridGamePredictor(game, game.modeler);
//		
//		ModuleManagerImpure mmm = new ModuleManagerImpure(0.9, 0.8);
//		int processTimes = 3;
//		mmm.processFullModel(game.modeler, relMngr, trainTurns, processTimes);
//		mmm.report();
//		
//		ModelLearnerModularImpure reuseModel = new ModelLearnerModularImpure(game.modeler, relMngr, mmm, trainTurns, processTimes);
//		reuseModel.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
//		game.modeler = reuseModel;
//
//		mmm.report();
//		
//		ReusableNNModule mod = mmm.getModuleUsedBy(game.modeler.getModelVTA().getNeuralNetwork().getOutputNodes().get(10));
//		TransferTestUtils.reportWeightsToOutput(output0);
//		System.out.println("---");
//		TransferTestUtils.reportWeightsToOutput(output5);
//		System.out.println("---");
//		TransferTestUtils.reportWeightsToOutput(output10);
//		System.out.println("---");
//		TransferTestUtils.reportWeightsToOutput(mod.getNeuralNet().getOutputNodes().get(0));
		
		
		
		
		FlierCatcher game = new FlierCatcher(size + 2);
		ModelLearnerHeavy dummyModel = T5GeoNet.makeModelerGeoNet(game, trainTurns, new int[] {3});
		Collection<TransitionMemory> trainingData = dummyModel.getExperience().getBatch();
		TransitionMemory tm0 = trainingData.iterator().next();
		RelationManager<Integer> relMngr = RelationManager.createFromGridGamePredictor(game,
				tm0.getPreStateAndAction().length, tm0.getPostState().length);
		int[] hiddenPerOutput = new int[] {30};
		int samplesPerTrain = 1500;
		int maxModules = 15;
		int maxModVar = 5;
		double killPct = 0.2;
		ModuleManagerPure moduleManager = new ModuleManagerPure(0.95, maxModules, killPct , 0.3, 0.5, 0, hiddenPerOutput, 200);
		ModelLearnerModularPure mlmp = new ModelLearnerModularPure(relMngr, moduleManager, samplesPerTrain, 3);
		game.modeler = mlmp;
		mlmp.saveMemories(trainingData);
		mlmp.learnGradually(maxModules-maxModVar, maxModules+maxModVar, 0.9, 0.98, 3);
		moduleManager.report();
		ModuleDisplayer.create(mlmp, tm0.getPostState().length, game);
		
	}
	
	private static double testAbstractPure(GameFactory srcFact, GameFactory trgFact, int hlSizeSrc, int hlSizeTrg,
			int trgTrainSamples, StringBuilder sb, int n) {
//		BiasNode.clearConnections();
		
		FlierCatcher sourceGame = srcFact != null ? srcFact.create() : new FlierCatcher(size);
		sourceGame.modeler = T6BasicWeightSharing.createModelerWithWgtSharing(sourceGame, hlSizeSrc, trainTurns, true);
		T5GeoNet.makeModelerGeoNet(sourceGame, trainTurns, new int[] {3});
		sourceGame.modeler.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
		final double score = sourceGame.modeler.getPctMastered();
		System.out.println("score " + score);
		sb.append("Source	" + score);
		FFNeuralNetwork vtaNN = sourceGame.modeler.getTransitionsModule().getNeuralNetwork();
		int inN = vtaNN.getInputNodes().size();
		int outN = vtaNN.getOutputNodes().size();
		RelationManager<Integer> catcherRelMngr = RelationManager.createFromGridGamePredictor(sourceGame, inN, outN);
		
		int maxModules1 = 10;
		int trainingEpochs = 50;
		ModuleManagerPure mmm = new ModuleManagerPure(0.98, maxModules1, 0.15, 0.3, 0.5, 0, new int[] {20}, trainingEpochs);
		int processTimes = 3;
		mmm.processFullModel(sourceGame.modeler, catcherRelMngr, trainTurns, processTimes);
		mmm.report();
		
		FlierCatcher targetGame = trgFact != null ? trgFact.create() : new FlierCatcher(size);
		ModelLearnerHeavy fullModel = T5GeoNet.makeModelerGeoNet(targetGame, trainTurns, new int[] {3});
		fullModel.learnFromMemory(0, 0, 0, false, 1, 10000);

		vtaNN = fullModel.getTransitionsModule().getNeuralNetwork();
		inN = vtaNN.getInputNodes().size();
		outN = vtaNN.getOutputNodes().size();
		RelationManager<Integer> fullRelMngr = RelationManager.createFromGridGamePredictor(targetGame, inN, outN);
		ModelLearnerModularPure reuseModel = new ModelLearnerModularPure(fullRelMngr, mmm, trgTrainSamples, processTimes);
		targetGame.modeler = reuseModel;
		reuseModel.saveMemories(fullModel.getExperience().getBatch());
		int maxModules2 = 60;
		reuseModel.learnGradually(maxModules1, maxModules2, 0.9, 0.99, 3);
//		targetGame.modeler = reuseModel;

		mmm.report();
		
		double transferScore = 0;
		for (int i = 0; i < n; i++) {
			transferScore += TransferTestUtils.compareTwoModelers(targetGame, 100, fullModel, reuseModel, true);
		}
		transferScore /= n;
		sb.append("	Transfer	" + transferScore + '\n');

//		ModelDisplayer modelDisplayer1 = new ModelDisplayer(fullModel, 2, targetGame);
//		ModelDisplayer modelDisplayer2 = new ModelDisplayer(reuseModel, 2, targetGame);
		ModuleDisplayer.create(reuseModel, outN, targetGame);
		
		return transferScore;
	}
	
	private static double testTransferGridSize(StringBuilder sb, int trgSampleSize, int numPlays) {
		return testAbstractPure(new GameFactory() {
			@Override
			public FlierCatcher create() {
				FlierCatcher game = new FlierCatcher(size);
				game.setSpawnFreq(0);
				return game;
			}
		}, new GameFactory() {
			@Override
			public FlierCatcher create() {
				FlierCatcher game = new FlierCatcher(size+2);
				return game;
			}
		}, size*2, size*size*3, trgSampleSize, sb, numPlays);
	}

	private static interface GameFactory {
		FlierCatcher create();
	}
}

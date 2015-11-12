package transfertests;

import ann.BiasNode;
import ann.FFNeuralNetwork;
import ann.Node;
import ann.indirectencodings.RelationManager;
import ann.testing.FlierCatcher;
import modeler.ModelLearner;
import modeler.ModelLearnerHeavy;
import modeler.ModelLearnerModularImpure;
import modeler.ModelLearnerModularPure;
import modulemanagement.ModelDisplayer;
import modulemanagement.ModuleDisplayer;
import modulemanagement.ModuleManagerImpure;
import modulemanagement.ModuleManagerPure;
import modulemanagement.ReusableNNModule;

public class T7HyperModelTest {
	static int size = 5;
	static int trainTurns = 1000;
	static int learnIterations = 100;
	
	public static void main(String[] args) {
//		testModules();
//		sanityCheck();
		testTransferGridSize(new StringBuilder(), 100, 4);
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
		FlierCatcher game = new FlierCatcher(size);
//		game.setSpawnFreq(0);
//		game.modeler = T6BasicWeightSharing.createModelerWithWgtSharing(game, size*2, trainTurns, true);
		T5GeoNet.makeModelerGeoNet(game, trainTurns, new int[] {3});
		System.out.println("Training source...");
		game.modeler.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
		Node output0 = game.modeler.getModelVTA().getNeuralNetwork().getOutputNodes().get(0);
		Node output5 = game.modeler.getModelVTA().getNeuralNetwork().getOutputNodes().get(5);
		Node output10 = game.modeler.getModelVTA().getNeuralNetwork().getOutputNodes().get(10);
		RelationManager<Node> relMngr = RelationManager.createFromGridGamePredictor(game, game.modeler);
		
		ModuleManagerImpure mmm = new ModuleManagerImpure(0.9, 0.8);
		int processTimes = 3;
		mmm.processFullModel(game.modeler, relMngr, trainTurns, processTimes);
		mmm.report();
		
		ModelLearnerModularImpure reuseModel = new ModelLearnerModularImpure(game.modeler, relMngr, mmm, trainTurns, processTimes);
		reuseModel.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
		game.modeler = reuseModel;

		mmm.report();
		
		ReusableNNModule mod = mmm.getModuleUsedBy(game.modeler.getModelVTA().getNeuralNetwork().getOutputNodes().get(10));
		TransferTestUtils.reportWeightsToOutput(output0);
		System.out.println("---");
		TransferTestUtils.reportWeightsToOutput(output5);
		System.out.println("---");
		TransferTestUtils.reportWeightsToOutput(output10);
		System.out.println("---");
		TransferTestUtils.reportWeightsToOutput(mod.getNeuralNet().getOutputNodes().get(0));
	}
	
	private static double testAbstractImpure(GameFactory srcFact, GameFactory trgFact, int hlSizeSrc, int hlSizeTrg,
			int trgTrainSamples, StringBuilder sb, int n) {
		BiasNode.clearConnections();
		
		FlierCatcher sourceGame = srcFact != null ? srcFact.create() : new FlierCatcher(size);
		sourceGame.modeler = T6BasicWeightSharing.createModelerWithWgtSharing(sourceGame, hlSizeSrc, trainTurns, true);
//		game.modeler = T6BasicWeightSharing.createModelerWithWgtSharing(game, size*2, trainTurns, true);
		T5GeoNet.makeModelerGeoNet(sourceGame, trainTurns, new int[] {3});
		sourceGame.modeler.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
		final double score = sourceGame.modeler.getPctMastered();
		System.out.println("score " + score);
		sb.append("Source	" + score);
		RelationManager<Node> catcherRelMngr = RelationManager.createFromGridGamePredictor(sourceGame, sourceGame.modeler);
		
		ModuleManagerImpure mmm = new ModuleManagerImpure(0.9, 0.8);
		int processTimes = 3;
		mmm.processFullModel(sourceGame.modeler, catcherRelMngr, trainTurns, processTimes);
		mmm.report();
		
		FlierCatcher targetGame = trgFact != null ? trgFact.create() : new FlierCatcher(size);
//		ModelLearnerHeavy fullModel = T6BasicWeightSharing.createModelerWithWgtSharing(targetGame, hlSizeTrg, trgTrainSamples, true);
		ModelLearnerHeavy fullModel = T5GeoNet.makeModelerGeoNet(targetGame, trainTurns, new int[] {3});
		fullModel.learnFromMemory(0, 0, 0, false, 1, 10000);
		RelationManager<Node> fullRelMngr = RelationManager.createFromGridGamePredictor(targetGame, fullModel);
		ModelLearnerModularImpure reuseModel = new ModelLearnerModularImpure(fullModel, fullRelMngr, mmm, trgTrainSamples, processTimes);
		reuseModel.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
		targetGame.modeler = reuseModel;

		mmm.report();
		
		double transferScore = 0;
		for (int i = 0; i < n; i++) {
			transferScore += TransferTestUtils.compareTwoModelers(targetGame, 100, fullModel, reuseModel, true);
		}
		transferScore /= n;
		sb.append("	Transfer	" + transferScore + '\n');

		ModelDisplayer modelDisplayer1 = new ModelDisplayer(fullModel, 2, targetGame);
		ModelDisplayer modelDisplayer2 = new ModelDisplayer(reuseModel, 2, targetGame);
		ModuleDisplayer moduleDisplayer = new ModuleDisplayer(reuseModel, 2, targetGame);
		
		return transferScore;
	}

	private static double testAbstractPure(GameFactory srcFact, GameFactory trgFact, int hlSizeSrc, int hlSizeTrg,
			int trgTrainSamples, StringBuilder sb, int n) {
		BiasNode.clearConnections();
		
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
		
		ModuleManagerPure mmm = new ModuleManagerPure(20, 0.3, 0.5, 0);
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
		reuseModel.saveMemories(fullModel.getExperience().getBatch());
		reuseModel.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
//		targetGame.modeler = reuseModel;

		mmm.report();
		
		double transferScore = 0;
		for (int i = 0; i < n; i++) {
			transferScore += TransferTestUtils.compareTwoModelers(targetGame, 100, fullModel, reuseModel, true);
		}
		transferScore /= n;
		sb.append("	Transfer	" + transferScore + '\n');

		return transferScore;
	}
	
	private static double testCatcherOnlyToFliers(StringBuilder sb, int trgSampleSize, int numPlays) {
		// src hidden layer size smaller cuz its just the catcher no fliers
		return testAbstractImpure(new GameFactory() {
			@Override
			public FlierCatcher create() {
				FlierCatcher game = new FlierCatcher(size);
				game.setSpawnFreq(0);
				return game;
			}
		}, null, size*size, size*size*3, trgSampleSize, sb, numPlays);
	}

	/** this shouldn't work well without module manager handling rotation translation intelligently 
	 * but also should be smart enough to not do significantly worse than scratch */
	private static double testCatcherOnlyToRotatedFliers(StringBuilder sb, int trgSampleSize, int numPlays) {
		// src hidden layer size smaller cuz its just the catcher no fliers
		return testAbstractImpure(new GameFactory() {
			@Override
			public FlierCatcher create() {
				FlierCatcher game = new FlierCatcher(size);
				game.setSpawnFreq(0);
				return game;
			}
		}, new GameFactory() {
			@Override
			public FlierCatcher create() {
				FlierCatcher game = new FlierCatcher(size);
				game.toggleRotation();
				return game;
			}
		}, size*2, size*size*3, trgSampleSize, sb, numPlays);
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

	private static void testModules() {
		FlierCatcher game = new FlierCatcher(size);
		ModelLearnerHeavy modelerHeavy = T6BasicWeightSharing.createModelerWithWgtSharing(game, size*size*2, trainTurns, true);
		RelationManager relMngr = RelationManager.createFromGridGamePredictor(game, modelerHeavy);
		modelerHeavy.learnFromMemory(0.9, 0.5, 0, false, learnIterations, 10000);
		
		ModuleManagerImpure mmm = new ModuleManagerImpure(0.99, 0.9);
		int processTurns = 200;
		int processTimes = 3;
		mmm.processFullModel(modelerHeavy, relMngr, processTurns, processTimes);
		
		// compare module-based modeler versus standard modeler
		ModelLearnerModularImpure modelerModular = new ModelLearnerModularImpure(modelerHeavy, relMngr, mmm, processTurns, processTimes);
		mmm.processFullModel(modelerModular, relMngr, processTurns, processTimes);
		mmm.report();

		TransferTestUtils.compareTwoModelers(game, 100, modelerHeavy, modelerModular, false);
	}
	
	private static void testDebug() {
		FlierCatcher game = new FlierCatcher(size);
		ModelLearner model = T6BasicWeightSharing.createModelerWithWgtSharing(game);
		model.learnFromMemory(0.9, 0.5, 0, false, learnIterations, 10000);
		
		RelationManager relMngr = RelationManager.createFromGridGamePredictor(game, model);
		System.out.println();
	}
	
	private static interface GameFactory {
		FlierCatcher create();
	}
}

package transfertests;

import ann.indirectencodings.RelationManager;
import ann.testing.FlierCatcher;
import modeler.ModelLearner;
import modeler.ModelLearnerHeavy;
import modeler.ModelLearnerModular;
import modularization.ModelModuleManager;

public class T7HyperModelTest {
	static int size = 5;
	static int trainTurns = 500;
	static int learnIterations = 100;
	
	public static void main(String[] args) {
//		testModules();
		testCatcherOnlyToFliers();
	}

	private static void testCatcherOnlyToFliers() {
		learnIterations = 200;
		
		FlierCatcher catcher = new FlierCatcher(size);
		catcher.setSpawnFreq(0);
		ModelLearnerHeavy catcherModel = T6BasicWeightSharing.createModelerWithWgtSharing("T7CATCHER", catcher, trainTurns, true);
		catcherModel.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
		RelationManager catcherRelMngr = RelationManager.createFromGridGamePredictor(catcher, catcherModel);
		
		ModelModuleManager mmm = new ModelModuleManager(catcher, 0.08, 0.1);
		int processTimes = 3;
		mmm.processFullModel(catcherModel, catcherRelMngr, trainTurns, processTimes);
		mmm.report();
		
		FlierCatcher flierCatcher = new FlierCatcher(size);
		int fewTrainTurns = 50; // sample starvation
		ModelLearnerHeavy fullModel = T6BasicWeightSharing.createModelerWithWgtSharing("T7FLIERS", flierCatcher, fewTrainTurns, true);
		fullModel.learnFromMemory(0, 0, 0, false, 1, 10000);
		RelationManager fullRelMngr = RelationManager.createFromGridGamePredictor(flierCatcher, fullModel);
		ModelLearnerModular reuseModel = new ModelLearnerModular(fullModel, fullRelMngr, mmm, fewTrainTurns, processTimes);
		reuseModel.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
		
		mmm.report();
		
		for (int i = 0; i < 8; i++) TransferTestUtils.compareTwoModelers(flierCatcher, 100, fullModel, reuseModel, true);

	}
	private static void testTransferGridSize() {
		FlierCatcher lilGame = new FlierCatcher(size);
		ModelLearnerHeavy lilModel = T6BasicWeightSharing.createModelerWithWgtSharing("T7LILSOURCE", lilGame, trainTurns, true);
		lilModel.learnFromMemory(0.9, 0.5, 0, false, learnIterations, 10000);
		RelationManager lilRelMngr = RelationManager.createFromGridGamePredictor(lilGame, lilModel);
//		Utils.saveModelerToFile("T7LILSOURCE", lilModel);
		
		ModelModuleManager mmm = new ModelModuleManager(lilGame, 0.01, 0.1);
		int processTimes = 3;
		mmm.processFullModel(lilModel, lilRelMngr, trainTurns, processTimes);
		mmm.report();
		
		FlierCatcher bigGame = new FlierCatcher(size + 1);
		int bigGameTrainTurns = 50; // sample starvation
		ModelLearnerHeavy bigModel = T6BasicWeightSharing.createModelerWithWgtSharing("T7BIGTARGET", bigGame, bigGameTrainTurns, true);
		bigModel.learnFromMemory(0, 0, 0, false, 1, 10000);
		RelationManager bigRelMngr = RelationManager.createFromGridGamePredictor(bigGame, bigModel);
		ModelLearnerModular reuseModel = new ModelLearnerModular(bigModel, bigRelMngr, mmm, bigGameTrainTurns, processTimes);
		reuseModel.learnFromMemory(0.9, 0.5, 0, false, learnIterations, 10000);
		
		mmm.report();
		
		for (int i = 0; i < 8; i++) TransferTestUtils.compareTwoModelers(bigGame, 100, bigModel, reuseModel, true);
	}

	private static void testModules() {
		FlierCatcher game = new FlierCatcher(size);
		ModelLearnerHeavy modelerHeavy = T6BasicWeightSharing.createModelerWithWgtSharing("T7MODULES", game, trainTurns, true);
		RelationManager relMngr = RelationManager.createFromGridGamePredictor(game, modelerHeavy);
		modelerHeavy.learnFromMemory(0.9, 0.5, 0, false, learnIterations, 10000);
		
		ModelModuleManager mmm = new ModelModuleManager(game, 0.01, 0.1);
		int processTurns = 200;
		int processTimes = 3;
		mmm.processFullModel(modelerHeavy, relMngr, processTurns, processTimes);
		
		// compare module-based modeler versus standard modeler
		ModelLearnerModular modelerModular = new ModelLearnerModular(modelerHeavy, relMngr, mmm, processTurns, processTimes);
		mmm.processFullModel(modelerModular, relMngr, processTurns, processTimes);
		mmm.report();

		TransferTestUtils.compareTwoModelers(game, 100, modelerHeavy, modelerModular, false);
	}
	
	private static void testDebug() {
		FlierCatcher game = new FlierCatcher(size);
		ModelLearner model = T6BasicWeightSharing.createModelerWithWgtSharing("T7DEBUG", game);
		model.learnFromMemory(0.9, 0.5, 0, false, learnIterations, 10000);
		
		RelationManager relMngr = RelationManager.createFromGridGamePredictor(game, model);
		System.out.println();
	}
	
}

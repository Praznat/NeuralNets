package transfertests;

import ann.Utils;
import ann.testing.FlierCatcher;
import ann.testing.FlierCatcher.MoveRule;
import modeler.ModelLearnerHeavy;
import modularization.WeightPruner;
import modulemanagement.ModelDisplayer;

public class T3IsolateIndependentRegions {
	final static int size = 5;
	final static int trainTurns = 500;
	final static int repaintMs = 50;
	final static int learnIterations = 100;
	
	public static void main(String[] args) {
		test1();
	}
	
	private static void test1() {
		FlierCatcher game = new FlierCatcher(size);
		game.setMoveRule(MoveRule.NORMAL);
		ModelLearnerHeavy model = TransferTestUtils.loadOrCreate("T3NORM", game, trainTurns);
		model.learnFromMemory(0.5, 0.5, 0, false, learnIterations , 10000);
		Utils.saveModelerToFile("T3NORM", model);
		
		modularize(model, 0.9, learnIterations);
		int[] gridSpec = new int[] {game.rows, game.cols};
		ModelDisplayer moduleDisplayer = new ModelDisplayer(model, 2, gridSpec, gridSpec, new int[] {4, 1});
		
	}
	
	static void modularize(ModelLearnerHeavy model, double pctileToPrune, int relearnIterations) {
//		WeightPruner.inOutAbsConnWgt(game.modeler.getModelJDM().getNeuralNetwork());
		WeightPruner.pruneBottomPercentile(model.getModelVTA().getNeuralNetwork(), pctileToPrune);
		WeightPruner.pruneBottomPercentile(model.getModelJDM().getNeuralNetwork(), pctileToPrune);
		model.getModelJDM().toggleShouldDisconnect(false);
		model.learnFromMemory(1.5, 0.5, 0, false, relearnIterations, 10000);
	}
}

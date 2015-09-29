package transfertests;

import ann.testing.FlierCatcher;
import ann.testing.FlierCatcher.MoveRule;
import modeler.ModelLearner;
import transfer.ModelSelector;
import transfer.SimulationScore;

public class T1FindBestModel {
	final static int size = 5;
	final static int trainTurns = 500;
	final static int testTurns = 5;
	final static int repaintMs = 50;
	final static int learnIterations = 100;

	public static void main(String[] args) {
		testFindSameGame();
	}
	
	public static void testFindSameGame() {
		FlierCatcher fcNorm = new FlierCatcher(size);
		fcNorm.setMoveRule(MoveRule.NORMAL);
		FlierCatcher fcCell = new FlierCatcher(size);
		fcCell.setMoveRule(MoveRule.CELLAUTONAMA);

		ModelLearner normModel = TransferTestUtils.loadOrCreate("T1NORM", fcNorm, trainTurns);
		normModel.learnFromMemory(0.5, 0.5, 0, false, learnIterations , 10000);
		ModelLearner cellModel = TransferTestUtils.loadOrCreate("T1CELL", fcCell, trainTurns);
		cellModel.learnFromMemory(0.5, 0.5, 0, false, learnIterations, 10000);

		ModelSelector selecta = new ModelSelector();
		
		selecta.loadModeler("T1NORM", normModel);
		selecta.loadModeler("T1CELL", cellModel);
			
		SimulationScore ss = selecta.observeWorkingModelTransitions(fcCell.modeler, testTurns, 20);
		System.out.println(ss);
	}
	

	public static void testTrainPartialGame() {
		
	}


}


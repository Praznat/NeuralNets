package transfertests;

import ann.indirectencodings.RelationManager;
import ann.testing.FlierCatcher;
import modeler.ModelLearner;

public class T7HyperModelTest {
	final static int size = 5;
	final static int trainTurns = 500;
	final static int learnIterations = 200;
	
	public static void main(String[] args) {
		testDebug();
	}

	private static void testDebug() {
		FlierCatcher game = new FlierCatcher(size);
		ModelLearner model = T6BasicWeightSharing.createModelerWithWgtSharing("T76DEBUG", game);
		model.learnFromMemory(0.9, 0.5, 0, false, learnIterations, 10000);
		
		RelationManager relMngr = RelationManager.createFromGridGamePredictor(game, model);
		System.out.println();
	}
	
}

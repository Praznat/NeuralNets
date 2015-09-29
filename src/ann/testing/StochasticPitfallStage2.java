package ann.testing;

import java.awt.*;
import java.util.Iterator;

import ann.*;
import modeler.EnvTranslator;
import reasoner.*;
import reasoner.DecisionProcess.LogLevel;

public class StochasticPitfallStage2 extends StochasticPitfall {

	private static final String SAVE_NAME = ""; // DOESNT SAVE JDM!
	private int currGeiser;
	private double painfullness = 20;
	private boolean isStochastic;
	
	protected RewardFunction MOVE = new RewardFunction() {
		@Override
		public double getReward(double[] stateVars) {
			final int k = rows * cols;
			double sumPain = 0;
			double sumGain = 0;
			for (int i = 0; i < (stateVars.length) / 2; i++) {
				final int c = i % cols;
				final int r = i / cols;
				sumPain -= stateVars[i] * stateVars[k + c + r*cols];
				sumGain += stateVars[i] * c;
			}
			return sumGain + sumPain * painfullness;
		}
	};

	// TODO VERSUS DETERMINISTIC!!!
	public static void main(String[] args) {
		for (Iterator<double[]> iter = ACTION_CHOICES.iterator(); iter.hasNext();) {
			double[] act = iter.next();
			if (act[1] > 0) iter.remove(); // no jumping
		}
		test();
		System.out.println("All done");
	}
	
	private static void test() {
		int size = 6;
		int learnIterations = 20;
		int sampleSizeMultiplier = 200;
		int repaintMsTraining = 10;
		int repaintMsTest = 100;
		boolean isStochastic = true;
		StochasticPitfallStage2 game = trainedGame(size, learnIterations,
				sampleSizeMultiplier, repaintMsTraining, isStochastic);
		System.out.println("Hit rate:	" + game.getScore());
		game.resetPoints();
		int numSteps = 1;
		int numRuns = ACTION_CHOICES.size() * 12;
		double discountRate = 0.0;
		int joints = 10;
		int upPainfulnessTurns = 50;
		double painChange = -3;
		Planner planner = Planner.createMonteCarloPlanner(game.modeler, numSteps, numRuns,
				game.MOVE, false, discountRate, joints, null, GridExploreGame.actionTranslator);
		double confusion = 0.01;
		double skewFactor = 0.05;
		double discRate = 0.1;
		double cutoffProb = 0.05;
		DecisionProcess decisionProcess = new DecisionProcess(game.modeler, ACTION_CHOICES, numSteps,
				numRuns, joints, skewFactor, discRate, cutoffProb);
		decisionProcess.setLogging(LogLevel.HI);
		for (int i = 0; i < upPainfulnessTurns * 10; i++) {
			if (i % upPainfulnessTurns == 0) {
				game.painfullness += painChange;
				game.setPlayerPos(0, game.floorRow());
				int numG = game.cols - game.cols/3;
				double pSurv = 1 - 1.0 / numG;
				System.out.println(numG + " geysers exist");
				System.out.println("Expected cost of walking is: "
						+ game.painfullness * (1 - Math.pow(pSurv, numG)));
				System.out.println("Expected benefit of walking is: " + ((game.cols+1)*game.cols)/2);
			}
			long startMs = System.currentTimeMillis();
			double[] preState = game.getState();
			game.chosenAction = planner.getOptimalAction(preState, ACTION_CHOICES, 0.01, 0.1);
//			game.chosenAction = decisionProcess.buildDecisionTree(preState, game.MOVE, numSteps, 0,
//					true, confusion);
			game.oneTurn();
			try {
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, repaintMsTest - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("Hit rate:	" + game.getScore());
	}
	
	public static StochasticPitfallStage2 trainedGame(int size, int learnIterations,
			int sampleSizeMultiplier, int repaintMs, boolean isStochastic) {
		StochasticPitfallStage2 game = new StochasticPitfallStage2(size);
		FFNeuralNetwork storedNet = Utils.loadNetworkFromFile(SAVE_NAME);
		game.isStochastic = isStochastic;
		game.modeler = trainedModeler(size * HEIGHT, size, game, sampleSizeMultiplier, repaintMs, ACTION_CHOICES,
				EnvTranslator.SAME, new int[] {size * 2});
		if (storedNet == null) {
			game.modeler.learnFromMemory(1.5, 0.5, 0, false, learnIterations, 10000);
			Utils.saveNetworkToFile(SAVE_NAME, game.modeler.getModelVTA().getNeuralNetwork());
		} else {
			game.modeler.getModelVTA().setANN(storedNet);
		}
		return game;
	}
	
	public StochasticPitfallStage2(int cols) {
		super(cols);
	}
	
	public void oneTurn() {
		doGeisers();
		super.oneTurn();
	}
	
	protected void doGeisers() {
		int start = cols/3;
		int newGeiser = isStochastic ? start + (int) (Math.random() * (cols - start))
				: (currGeiser - start + 1) % (cols - start) + start;
		for (int r = 0; r < rows; r++) {
			opponentGrid[currGeiser][r] = 0;
			opponentGrid[newGeiser][r] = 1;
		}
		currGeiser = newGeiser;
	}
	
	protected void paintObstacle(Graphics g, int c, int r, int gSub) {
		g.setColor(Color.WHITE);
		g.fillRect(c*gSub + gSub/4, r*gSub, gSub - gSub/2, gSub);
	}
	
	protected void morePainting(Graphics g, int gSub) {
		g.setColor(Color.BLACK);
		g.drawString("Painfulness: " + painfullness, gSub, gSub);
	}
	
}

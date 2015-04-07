package deepnets.testing;

import java.awt.*;

import modeler.EnvTranslator;
import reasoner.*;

public class StochasticPitfallStage1 extends StochasticPitfall {

	protected double logFrequency;
	
	protected RewardFunction AVOID_LOGS = new RewardFunction() {
		@Override
		public double getReward(double[] stateVars) {
			final int k = rows * cols;
			double sum = 0;
			for (int i = 0; i < stateVars.length / 2; i++) {
				final int c = i % cols;
				final int r = i / cols;
				sum -= stateVars[i] * stateVars[k + c + r*cols];
			}
			return sum;
		}
	};

	public static void main(String[] args) {
		test();
		System.out.println("All done");
	}
	
	// TODO override action when jumping to incorporate past action into Markovian learner
	private static void test() {
		int size = 8;
		double logFrequency = 0.3;
		int learnIterations = 150;
		int sampleSizeMultiplier = 100;
		int repaintMsTraining = 10;
		int repaintMsTest = 100;
		StochasticPitfallStage1 game = trainedGame(size, logFrequency, learnIterations,
				sampleSizeMultiplier, repaintMsTraining);
		System.out.println("Hit rate:	" + game.getScore());
		game.resetPoints();
		int numSteps = 6;
		int numRuns = ACTION_CHOICES.size() * 3;
		int joints = 10;
		Planner planner = Planner.createMonteCarloPlanner(game.modeler, numSteps, numRuns, game.AVOID_LOGS,
				false, 0.25, joints, null, GridExploreGame.actionTranslator);
		for (int i = 0; i < 1000; i++) {
			long startMs = System.currentTimeMillis();
			double[] preState = game.getState();
			game.chosenAction = planner.getOptimalAction(preState, ACTION_CHOICES, 0.01, 0.1);
			game.setupForTurn();
			// TODO if jumping, override chosenAction with last action taken on ground
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
	
	public static StochasticPitfallStage1 trainedGame(int size, double logFrequency,
			int learnIterations, int sampleSizeMultiplier, int repaintMs) {
		StochasticPitfallStage1 game = new StochasticPitfallStage1(size, logFrequency);
		game.modeler = trainedModeler(size * HEIGHT, size, game, sampleSizeMultiplier, repaintMs, ACTION_CHOICES,
				EnvTranslator.SAME, null);
		game.modeler.learnFromMemory(1.5, 0.5, 0, false, learnIterations, 10000);
		return game;
	}
	
	public StochasticPitfallStage1(int cols, double logFrequency) {
		super(cols);
		this.logFrequency = logFrequency;
	}
	
	public void oneTurn() {
		moveLogs();
		super.oneTurn();
	}
	
	public void setupForTurn() {
		if (isPlayerJumping()) {
			chosenAction = oldAction;
		}
	}
	
	protected void moveLogs() {
		int fr = floorRow();
		for (int c = 0; c < cols-1; c++) {
			opponentGrid[c][fr] = opponentGrid[c+1][fr];
		}
		opponentGrid[cols-1][fr] = 0;
		if (Math.random() < logFrequency && opponentGrid[cols-2][fr] < 0.5) // no adjacent logs
			opponentGrid[cols-1][fr] = 1;
	}
	
	private static Color BROWN = new Color(200,200,50);
	
	protected void paintObstacle(Graphics g, int c, int r, int gSub) {
		g.setColor(BROWN);
		g.fillOval(c*gSub, r*gSub, gSub, gSub);
	}
	
}

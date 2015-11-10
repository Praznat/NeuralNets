package ann.testing;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;

import ann.ActivationFunction;
import ann.BiasNode;
import ann.Utils;
import modeler.ModelLearnerHeavy;
import reasoner.Abduction;
import reasoner.Foresight;
import reasoner.Planner;
import reasoner.RewardFunction;
import utils.RandomUtils;

/**
 * we have a grid world with a player, an opponent, and walls
 * in mode 1, the opponent moves deterministically, and the player must learn to tag the opponent
 * in mode 2, the opponent moves deterministically eventually covering the whole grid,
 * and the player must learn to avoid the opponent
 * 
 * @author alexanderbraylan
 *
 */
public class GridTagGame extends GridGame {

	int numTags;
	{
		actionChoices.clear();
		actionChoices.add(new double[] {0}); //CLOCKWISE
		actionChoices.add(new double[] {1}); //COUNTERCLOCKWISE
	}
	
	public static void main(String[] args) {
//		testExplore();
//		testAbduction(true);
//		testAbduction(false);
		for (int i = 0; i < 10; i++) test2();
	}

	protected static void testAbduction(boolean clockwise) {
		int size = 3;
		int turns = 500;
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {size * 4 * 2},
				new int[] {}, new int[] {}, ActivationFunction.SIGMOID0p5, turns);
		GridTagGame game = new GridTagGame(size, size);
		game.setupGameDisplay();
		game.setPlayerRule(game.clockwiseAction);
		game.setOpponentRule(game.idle);
		game.setPlayerPos(size-1, size-1);
		game.setOpponentPos(0, 0);
		for (int t = 0; t < turns; t++) {
			modeler.observePreState(game.getState());
			game.chosenAction = RandomUtils.randomOf(game.actionChoices);
			modeler.observeAction(game.chosenAction);
			game.oneTurn();
			modeler.observePostState(game.getState());
			modeler.saveMemory();
		}
		modeler.learnFromMemory(1.5, 0.5, 0, false, 100);
		
		if (!clockwise) game.setPlayerPos(size-1, size-2);
		else game.setPlayerPos(size-2, size-1);
		double[] targetState = game.getState();
		game.setPlayerPos(size-1, size-1);
		double[] gameState = game.getState();
		double[] inVars = new double[gameState.length + game.chosenAction.length];
		System.arraycopy(gameState, 0, inVars, 0, gameState.length);
		inVars[inVars.length-1] = 0.5;
		double guessAction = Abduction.backPropAbduction(modeler, targetState, inVars,
				new int[] {inVars.length-1})[inVars.length-1];
		boolean correct = clockwise ? guessAction > 0.5 : guessAction < 0.5;
		System.out.println(correct ? " CORRECT" : "FAIL");
	}
	protected static void testReward() {
		int size = 3;
		GridTagGame game = new GridTagGame(size, size);
		game.setupGameDisplay();
		game.setPlayerPos(1, 1);
		game.setOpponentPos(1, 1);
		System.out.println(-1 + "==" + evade(game).getReward(game.getState()));
		game.setOpponentPos(0, 1);
		System.out.println(0 + "==" + evade(game).getReward(game.getState()));
	}
	

	protected static void testExplore() {
		int size = 3;
		int turns = 500;
		int numPlanSteps = 1;
		int numPlanRuns = 1;
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {size * 4 * 2},
				new int[] {}, new int[] {}, ActivationFunction.SIGMOID0p5, turns);
		Planner chimp = Planner.createRandomChimp();
		Planner explorer = Planner.createNoveltyExplorer(modeler, numPlanSteps, numPlanRuns, null, null);
		GridTagGame game = new GridTagGame(size, size);
		game.setupGameDisplay();
		game.setPlayerRule(game.clockwiseAction);
		game.setOpponentRule(game.idle);
		game.setPlayerPos(size-1, size-1);
		game.setOpponentPos(0, 0);
		for (int t = 0; t < turns; t++) {
			long startMs = System.currentTimeMillis();
			modeler.observePreState(game.getState());
			game.chosenAction = (t < turns / 4 ? chimp : explorer).getOptimalAction(game.getState(), game.actionChoices, 0, 0);
			modeler.observeAction(game.chosenAction);
			game.oneTurn();
			modeler.observePostState(game.getState());
			modeler.learnOnline(1.5, 0.5, 0); // instead of saveMemory
			try {
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, 100 - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	protected static void test1() {
		int size = 3;
		int turns = 500;
		Planner explorer = Planner.createRandomChimp();
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {size * 4 * 2},
				new int[] {}, new int[] {}, ActivationFunction.SIGMOID0p5, turns);
		GridTagGame game = new GridTagGame(size, size);
		game.setupGameDisplay();
		game.setPlayerRule(game.clockwiseAction);
		game.setOpponentRule(game.rightBounceOffWall);
		for (int i = 1; i < size - 1; i++) game.walls[i][1] = 1;
		game.setPlayerPos(size-1, size-1);
		game.setOpponentPos(0, 0);
		
		for (int t = 0; t < turns; t++) {
			modeler.observePreState(game.getState());
			game.chosenAction = explorer.getOptimalAction(game.getState(), game.actionChoices, 0, 0);
			modeler.observeAction(game.chosenAction);
			game.oneTurn();
			modeler.observePostState(game.getState());
			modeler.saveMemory();
			game.print(game.chosenAction);
		}
		modeler.learnFromMemory(1.5, 0.5, 0, false, 100);
		for (int t = 0; t < turns; t++) {
			game.print(game.chosenAction);
			modeler.observePreState(game.getState());
			game.chosenAction = explorer.getOptimalAction(game.getState(), game.actionChoices, 0, 0);
			modeler.observeAction(game.chosenAction);
			double[] prophesy = Foresight.montecarlo(modeler, game.getState(), game.chosenAction, null, 1, 10, 10, .1);
			// TODO get positions from prophesy so we can do accuracy measure
			print(game.getGridFromState(prophesy, true), game.getGridFromState(prophesy, false));
			modeler.observePostState(game.getState());
			game.oneTurn();
		}
	}
	protected static void test2() {
		int size = 3;
		int trainTurns = size * 4 * 200;
		int gameTurns = 500;
		int maxNumPlanSteps = size;
		int numPlanRuns = 25;
		int joints = 3;
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {size * 4 * 2},
				null, null, ActivationFunction.SIGMOID0p5, trainTurns);
//		Planner explorer = Planner.createNoveltyExplorer(modeler, numPlanSteps, numPlanRuns, null, null);
		Planner explorer = Planner.createRandomChimp();
		GridTagGame game = new GridTagGame(size, size);
		game.setupGameDisplay();
		game.setPlayerRule(game.clockwiseAction);
		game.setOpponentRule(game.mostlyClockwise);
		game.setPlayerPos(size-1, size-1);
		game.setOpponentPos(0, 0);
		BiasNode.clearConnections();
		
		for (int t = 0; t < trainTurns; t++) {
			game.chosenAction = explorer.getOptimalAction(game.getState(), game.actionChoices, 0, 0);
			modeler.observeAction(game.chosenAction);
			modeler.observePreState(game.getState());
			game.oneTurn();
			modeler.observePostState(game.getState());
			modeler.learnOnline(1.5, 0.5, 0); // instead of saveMemory
		}
		modeler.learnFromMemory(1.5, 0.5, 0, false, 150, 10000);
//		WeightPruner.pruneBelowAvg(modeler.getModelVTA().getNeuralNetwork(), 0.2);
//		modeler.learnFromMemory(1.5, 0.5, 0, false, 100, 10000);
//		WeightPruner.inOutAbsConnWgt(modeler.getModelVTA().getNeuralNetwork());
	
		// up to now should be similar test1 (except walls, exploration, etc)
		// TODO random mutate actions when both rewards are zero using +-
		// TODO learning to teleport ok?
		for (int numPlanSteps = 1; numPlanSteps <= maxNumPlanSteps; numPlanSteps++) {
			game.numTags = 0;
			game.setPlayerPos(size-1, size-1);
			game.setOpponentPos(0, 0);
			Planner follower = Planner.createMonteCarloPlanner(modeler, numPlanSteps, numPlanRuns, follow(game), true, 0.1, joints);
			Planner evader = Planner.createMonteCarloPlanner(modeler, numPlanSteps, numPlanRuns, evade(game), true, 0.1, joints);
			Planner planner = follower;
			int wins = 0; int losses = 0;
			for (int t = 0; t < gameTurns; t++) {
				long startMs = System.currentTimeMillis();
				if (t % 50 == 0) { // switch
					boolean isTag = planner == follower;
					if (isTag) wins += game.numTags;
					else losses += game.numTags;
					game.numTags = 0;
					game.isTag = !isTag;
					planner = (isTag ? evader : follower);
					game.resetPos();
				}
//				game.print(game.chosenAction);
				game.chosenAction = planner.getOptimalAction(game.getState(), game.actionChoices, 0, 0);
				
				double[] prophesy = Foresight.montecarlo(modeler, game.getState(), game.chosenAction,
					numPlanSteps, numPlanRuns, 10);
				print(game.getGridFromState(prophesy, true), game.getGridFromState(prophesy, false));
				
				try {
					long elapsedMs = System.currentTimeMillis() - startMs;
					Thread.sleep(Math.max(0, 100 - elapsedMs));
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				game.oneTurn();
			}
			System.out.println("STEPS:	" + numPlanSteps + "	WINS:	" + wins + "	LOSSES:	" + losses);
		}
	}

	final int[][] playerGrid; // 1 represents location of player, 0 otherwise
	final int[][] opponentGrid; // 1 represents location of opponent, 0 otherwise
	final int[][] walls; // these should not be input into modeler as they are static and derivable from transitions

	private void tag() {
		numTags++;
		// teleport player to opposite side!
		resetPos();
	}
	private void resetPos() {
		setPlayerPos(cols - opponentPos.x - 1, rows - opponentPos.y - 1);
	}
	protected boolean isInBoundsC(int c) {
		return c >= 0 && c < cols;
	}
	protected boolean isInBoundsR(int r) {
		return r >= 0 && r < rows;
	}
	protected Point rotate(Point from, boolean isClockWise) {
		int m = isClockWise ? 1 : -1;
		int c = from.x;
		int r = from.y;
		if (c == 0 && isInBoundsR(r - m)) return getGridLocPt(from.x, from.y-m);
		else if (r == 0 && isInBoundsC(c + m)) return getGridLocPt(from.x+m, from.y);
		else if (c == cols-1 && isInBoundsR(r + m)) return getGridLocPt(from.x, from.y+m);
		else if (r == rows-1 && isInBoundsC(c - m)) return getGridLocPt(from.x-m, from.y);
		else throw new IllegalStateException("BUG in rotate (illegal coordinates)");
	}
	final PlayerRule clockwiseAction = new PlayerRule() {
		@Override
		Point posTo(Point from, double[] action) {
			boolean isClockWise = action[0] > 0;
			return rotate(from, isClockWise);
		}
	};
	final OpponentRule mostlyClockwise = new OpponentRule() {
		final double explore = 0.99; // TODO set higher and see difference between joint and no joint
		@Override
		Point posTo(Point from) {
			boolean isClockWise = Math.random() >= explore;
			return rotate(from, isClockWise);
		}
	};
	final OpponentRule idle = new OpponentRule() {
		@Override
		Point posTo(Point from) {
			return from;
		}
	};
	final OpponentRule rightBounceOffWall = new OpponentRule() {
		final double explore = 0.3;
		@Override
		Point posTo(Point from) {
			int d = Math.random() < explore ? -1 : 1;
			boolean isWall = getGridLoc(from.x+d, from.y, walls) > 0;
			return isWall ? getGridLocPt(from.x-d, from.y) : getGridLocPt(from.x+d, from.y);
		}
	};

	Point playerPos = new Point();
	Point opponentPos = new Point();
	PlayerRule playerRule;
	OpponentRule opponentRule;
	RewardFunction rewardFunction;
	
	public GridTagGame(int rows, int cols) {
		super(rows, cols);
		playerGrid = new int[cols][rows];
		opponentGrid = new int[cols][rows];
		walls = new int[cols][rows];
	}
	
	public void oneTurn() {
		opponentRule.move();
		playerRule.move(chosenAction);
		repaint();
	}
	
	public double[][] getGridFromState(double[] state, boolean opponentNotPlayer) {
		double[][] result = new double[cols][rows];
		int k = opponentNotPlayer ? rows * cols : 0;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) result[c][r] = state[k + c + r*cols];
		}
		return result;
	}
	@Override
	public double[] getState() {
		double[] result = new double[rows * cols * 2];
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) result[c + r*cols] = playerGrid[c][r];
		}
		int k = rows * cols;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) result[k + c + r*cols] = opponentGrid[c][r];
		}
		return result;
	}
	
	protected void setPlayerPos(Point p) {
		setPlayerPos(p.x, p.y);
	}
	protected void setPlayerPos(int col, int row) {
		setPos(col, row, playerPos, playerGrid);
	}
	protected void setOpponentPos(Point p) {
		setOpponentPos(p.x, p.y);
	}
	protected void setOpponentPos(int col, int row) {
		setPos(col, row, opponentPos, opponentGrid);
	}
	protected void setPos(int col, int row, Point pos, int[][] grid) {
		grid[pos.x][pos.y] = 0;
		pos.setLocation((col + cols) % cols, (row + rows) % rows);
		grid[pos.x][pos.y] = 1;
	}
	protected void setPlayerRule(PlayerRule pr) {
		playerRule = pr;
	}
	protected void setOpponentRule(OpponentRule or) {
		opponentRule = or;
	}
	
	protected Point getGridLocPt(int col, int row) {
		return new Point((col + cols) % cols, (row + rows) % rows);
	}

	public static void print(double[][] ds) {
		for (int r = 0; r < ds[0].length; r++) {
			String s = "";
			for (int c = 0; c < ds.length; c++) s += Utils.round(ds[c][r], 4) + "	";
			System.out.println(s);
		}
		System.out.println("********************************");
	}
	public static void print(double[][] o, double[][] p) {
		for (int r = 0; r < o[0].length; r++) {
			String s = "";
			for (int c = 0; c < o.length; c++) s += Utils.round(o[c][r], 4) + "	";
			s += "	";
			for (int c = 0; c < p.length; c++) s += Utils.round(p[c][r], 4) + "	";
			System.out.println(s);
		}
		System.out.println("************************************************************");
	}
	public void print(double[] action) {
		for (int r = 0; r < rows; r++) {
			String s = "";
			for (int c = 0; c < cols; c++) {
				if (playerGrid[c][r] > 0) s += "p	";
				else if (opponentGrid[c][r] > 0) s += "e	";
				else if (walls[c][r] > 0) s += (action[0] > 0 ? "r" : "l") + "	";
				else s += ".	";
			}
			System.out.println(s);
		}
		System.out.println("***********");
	}

	static final RewardFunction tagReward(final boolean shouldTag, final int cols, final int rows) {
		return new RewardFunction() {
			@Override
			public double getReward(double[] stateVars) {
				final int k = rows * cols;
				double sum = 0;
				for (int i = 0; i < stateVars.length / 2; i++) {
					final int c = i % cols;
					final int r = i / cols;
					sum += stateVars[i] * stateVars[k + c + r*cols] * (shouldTag?1:-1);
				}
				return sum;
			}
		};
	}
	public static final RewardFunction follow(GridGame game) {
		return tagReward(true, game.cols, game.rows);
	}
	public static final RewardFunction evade(GridGame game) {
		return tagReward(false, game.cols, game.rows);
	}

	abstract class PlayerRule {
		abstract Point posTo(Point from, double[] action);
		void move(double[] action) {
			if (action != null) setPlayerPos(posTo(playerPos, action));
		}
	}
	abstract class OpponentRule {
		abstract Point posTo(Point from);
		void move() {
			setOpponentPos(posTo(opponentPos));
		}
	}

	private boolean isTag = true;
	private Color eColor() {return isTag? Color.GREEN : Color.RED;}
	
	@Override
	public void paintGrid(Graphics g) {
		final int gUnit = 100;
		final int gSub = 10;
		g.setColor(Color.BLACK);
		int[][] pGrid = this.playerGrid;
		boolean tag = false;
		for (int c = 0; c < pGrid.length; c++) {
			int[] col = pGrid[c];
			for (int r = 0; r < col.length; r++) {
				g.drawRect(c*gUnit, r*gUnit, gUnit, gUnit);
				boolean player = col[r] > 0;
				if (this.opponentGrid[c][r] > 0) {
					g.setColor(eColor());
					g.fillRect(c*gUnit, r*gUnit, gUnit, gUnit);
					if (player) {
						g.setColor(Color.BLUE);
						g.fillRect(c*gUnit, r*gUnit, gUnit, gUnit);
						tag = true;
					}
					g.setColor(Color.BLACK);
					g.drawLine(c*gUnit+gSub, r*gUnit+gSub, (c+1)*gUnit-gSub, (r+1)*gUnit-gSub);
					g.drawLine(c*gUnit+gSub, (r+1)*gUnit-gSub, (c+1)*gUnit-gSub, r*gUnit+gSub);
				}
				if (player) g.fillOval(c*gUnit+gSub, r*gUnit+gSub, gUnit - 2*gSub, gUnit - 2*gSub);
			}
		}
		if (tag) tag();
	}
}

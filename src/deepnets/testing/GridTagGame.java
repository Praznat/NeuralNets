package deepnets.testing;

import java.awt.*;
import java.util.*;
import java.util.List;

import modeler.ModelLearnerHeavy;
import reasoner.*;
import deepnets.*;

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

	protected static final List<double[]> ACTION_CHOICES = new ArrayList<double[]>();
	static {
		ACTION_CHOICES.add(new double[] {0});
		ACTION_CHOICES.add(new double[] {1});
	}
	
	public static void main(String[] args) {
		test2();
	}

	protected static void testReward() {
		int size = 3;
		GridTagGame game = new GridTagGame(size, size);
		gridPanel.setGame(game);
		game.setPlayerPos(1, 1);
		game.setOpponentPos(1, 1);
		System.out.println(-1 + "==" + game.evade.getReward(game.getState()));
		game.setOpponentPos(0, 1);
		System.out.println(0 + "==" + game.evade.getReward(game.getState()));
	}
	protected static void test1() {
		int size = 3;
		int turns = 500;
		Planner explorer = Planner.createRandomChimp();
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {size * size * 3},
				new int[] {}, new int[] {}, ActivationFunction.SIGMOID0p5, turns);
		GridTagGame game = new GridTagGame(size, size);
		gridPanel.setGame(game);
		game.setPlayerRule(game.clockwiseAction);
		game.setOpponentRule(game.rightBounceOffWall);
		for (int i = 1; i < size - 1; i++) game.walls[i][1] = 1;
		game.setPlayerPos(size-1, size-1);
		game.setOpponentPos(0, 0);
		
		for (int t = 0; t < turns; t++) {
			modeler.observePreState(game.getState());
			game.chosenAction = explorer.getOptimalAction(game.getState(), ACTION_CHOICES, 0, 0);
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
			game.chosenAction = explorer.getOptimalAction(game.getState(), ACTION_CHOICES, 0, 0);
			modeler.observeAction(game.chosenAction);
			double[] prophesy = Foresight.montecarlo(modeler, game.getState(), game.chosenAction, 1, 10, 10, .1);
			// TODO get positions from prophesy so we can do accuracy measure
			print(game.getGridFromState(prophesy, true), game.getGridFromState(prophesy, false));
			modeler.observePostState(game.getState());
			game.oneTurn();
		}
	}
	protected static void test2() {
		int size = 5;
		int turns = size * size * 50;
		Planner explorer = Planner.createRandomChimp();
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {size * size * 2},
				new int[] {}, new int[] {}, ActivationFunction.SIGMOID0p5, turns);
		GridTagGame game = new GridTagGame(size, size);
		gridPanel.setGame(game);
		game.setPlayerRule(game.clockwiseAction);
		game.setOpponentRule(game.mostlyClockwise);
		game.setPlayerPos(size-1, size-1);
		game.setOpponentPos(0, 0);
		
		for (int t = 0; t < turns; t++) {
			modeler.observePreState(game.getState());
			game.chosenAction = explorer.getOptimalAction(game.getState(), ACTION_CHOICES, 0, 0);
			modeler.observeAction(game.chosenAction);
			game.oneTurn();
			modeler.observePostState(game.getState());
			modeler.saveMemory();
		}
		modeler.learnFromMemory(1.5, 0.5, 0, false, 100);
		// up to now should be same as test1 (except walls)
		game.setPlayerPos(size-1, size-1);
		game.setOpponentPos(0, 0);
		int numPlanSteps = 3;
		int numPlanRuns = 25;
		Planner follower = Planner.createMonteCarloPlanner(modeler, numPlanSteps, numPlanRuns, game.follow);
		Planner evader = Planner.createMonteCarloPlanner(modeler, numPlanSteps, numPlanRuns, game.evade);
		Planner planner = follower;
		for (int t = 0; t < turns; t++) {
			long startMs = System.currentTimeMillis();
			if (t % 50 == 0) {
				boolean isTag = planner == follower;
				game.isTag = !isTag;
				planner = (isTag ? evader : follower);
			}
			game.print(game.chosenAction);
			modeler.observePreState(game.getState());
			game.chosenAction = planner.getOptimalAction(game.getState(), ACTION_CHOICES, 0, 0);
			modeler.observeAction(game.chosenAction);
			modeler.observePostState(game.getState());
			try {
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, 100 - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			game.oneTurn();
		}
	}

	final int[][] playerGrid; // 1 represents location of player, 0 otherwise
	final int[][] opponentGrid; // 1 represents location of opponent, 0 otherwise
	final int[][] walls; // these should not be input into modeler as they are static and derivable from transitions

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
		final double explore = 0.2;
		@Override
		Point posTo(Point from) {
			boolean isClockWise = Math.random() >= explore;
			return rotate(from, isClockWise);
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
	double[] chosenAction;
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
		frame.repaint();
	}
	
	public double[][] getGridFromState(double[] state, boolean opponentNotPlayer) {
		double[][] result = new double[cols][rows];
		int k = opponentNotPlayer ? rows * cols : 0;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) result[c][r] = state[k + c + r*cols];
		}
		return result;
	}
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

	RewardFunction getRewardFunction() {
		return rewardFunction;
	}
	void setRewardFunction(RewardFunction rewardFunction) {
		 this.rewardFunction = rewardFunction;
	}
	final RewardFunction follow = new RewardFunction() {
		@Override
		public double getReward(double[] stateVars) {
			final int k = rows * cols;
			for (int i = 0; i < stateVars.length / 2; i++) {
				final int c = i % cols;
				final int r = i / cols;
				if (stateVars[i] > .5 && stateVars[k + c + r*cols] > .5) return 1;
			}
			return 0;
		}
	};
	final RewardFunction evade = new RewardFunction() {
		// TODO maybe should be about proximity not exact overlap
		@Override
		public double getReward(double[] stateVars) {
			final int k = rows * cols;
			for (int i = 0; i < stateVars.length / 2; i++) {
				final int c = i % cols;
				final int r = i / cols;
				if (stateVars[i] > .5 && stateVars[k + c + r*cols] > .5) return -1;
			}
			return 0;
		}
	};

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

	boolean isTag = true;
	Color eColor() {return isTag? Color.GREEN : Color.RED;}
	
	@Override
	protected void paintGrid(Graphics g) {
		final int gUnit = 100;
		final int gSub = 10;
		g.setColor(Color.BLACK);
		int[][] pGrid = this.playerGrid;
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
					}
					g.setColor(Color.BLACK);
					g.drawLine(c*gUnit+gSub, r*gUnit+gSub, (c+1)*gUnit-gSub, (r+1)*gUnit-gSub);
					g.drawLine(c*gUnit+gSub, (r+1)*gUnit-gSub, (c+1)*gUnit-gSub, r*gUnit+gSub);
				}
				if (player) g.fillOval(c*gUnit+gSub, r*gUnit+gSub, gUnit - 2*gSub, gUnit - 2*gSub);
			}
		}
	}
}

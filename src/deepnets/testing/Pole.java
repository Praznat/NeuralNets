package deepnets.testing;

//Inverted Pendulum simulation
//by Chuck Anderson, 1998, with code from O'Reilly's "Java by Example"

import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;

import modeler.*;
import reasoner.*;
import deepnets.ActivationFunction;

public class Pole extends Applet implements Runnable {
	int delay;
	Thread animatorThread;
	//next three are for double-buffering
	Dimension offDimension;
	Image offImage;
	Graphics offGraphics;
	//Now for pole simulation
	int action;
	double pos, posDot, angle, angleDot;

	double smartThresh = 0.0001;
	static EnvTranslator actionTranslator = new EnvTranslator() {
		@Override
		public double[] toNN(double... n) {
			if (n[0] == 1) return new double[] {1,0};
			if (n[0] == -1) return new double[] {0,1};
			else return new double[] {0,0};
		}
		@Override
		public double[] fromNN(double[] d) {
			if (d[0] == 1) return new double[] {1.0};
			if (d[1] == 1) return new double[] {-1.0};
			else return new double[] {0.0};
		}
	};
	static double[] stateMins = new double[] {-1.5, -.12}; //{-1.2, -.025};
	static double[] stateMaxes = new double[] {1.5, .12}; //{1.2, .025};
	static final int NUM_BUCKETS = 8;
	static EnvTranslator stateTranslator = EnvTranslator.rbfEnvTranslator(stateMins,
			stateMaxes, new int[] {NUM_BUCKETS,NUM_BUCKETS}, .5);
	int learnFrame = 1000;
	final int learnFrameInc = learnFrame;
	final ModelLearnerHeavy modeler = new ModelLearnerHeavy(100, new int[] {NUM_BUCKETS*2},
			new int[] {NUM_BUCKETS*2*2}, new int[] {NUM_BUCKETS*2}, ActivationFunction.SIGMOID0p5, learnFrame);
	int frame = 0;
	final int planningSteps = 5;
	final Planner planner = Planner.createMonteCarloPlanner(modeler, planningSteps, 50, new RewardFunction() {
		@Override
		public double getReward(double[] stateVars) {
//			return -Math.abs(stateTranslator.fromNN(stateVars)[0]);
			return -stateVars[0] - stateVars[NUM_BUCKETS-1];
		}
	}, true, 0.1, 10, stateTranslator, actionTranslator);
	int lastLoss = 0;
	static final List<double[]> actionChoices = new ArrayList<double[]>();
	private boolean lastWasLoss = true;
	{
		actionChoices.add(new double[] {1,0});
		actionChoices.add(new double[] {0,0});
		actionChoices.add(new double[] {0,1});
	}
	private final Collection<double[]> tmpCorrel = new ArrayList<double[]>();
	
	//Constants
	public static final double cartMass=1.;
	public static final double poleMass=0.1;
	public static final double poleLength=1.; ;
	public static final double forceMag=20.;
	public static final double tau=0.02;
	public static final double fricCart=0.00005;
	public static final double fricPole=0.005;
	public static final double totalMass = cartMass + poleMass;
	public static final double halfPole = 0.5 * poleLength; ;
	public static final double poleMassLength = halfPole * poleMass;
	public static final double fourthirds = 4./3.; 


	public void init() {
		String str;
		int fps = 30;

		// Initialize pole state.
		initVars();

		// Set up animation timing.
		//How many milliseconds between frames?
		str = getParameter("fps");
		try {
			if (str != null) {
				fps = Integer.parseInt(str);
			}
		} catch (Exception e) {}
		delay = (fps > 0) ? (1000 / fps) : 100;


		// Event handlers
		this.addMouseListener(new MouseAdapter() {
			public void mousePressed(MouseEvent e) {
				if ((e.getModifiers() & e.BUTTON1_MASK) == e.BUTTON1_MASK) {
					action = -1;
				} else if ((e.getModifiers() & e.BUTTON2_MASK) == e.BUTTON2_MASK) {
					action = 0;
					resetPole();
				} else if ((e.getModifiers() & e.BUTTON3_MASK) == e.BUTTON3_MASK) {
					action = 1;
				}
			}
			public void mouseReleased(MouseEvent e) {
				action = 0;
				//	System.out.println("mouse is " + e.getModifiers());
			}

		});

		this.addKeyListener(new KeyAdapter() {
			public void keyPressed(KeyEvent e) {
				//System.out.println("keycode is " + e.getKeyCode() + " id " + e.getID());
				if (e.getKeyCode() == KeyEvent.VK_LEFT)
					action = -1;
				else if (e.getKeyCode() == KeyEvent.VK_RIGHT) 
					action = 1;
				else if (e.getKeyChar() == ' ') {
					action = 0;
					resetPole();
				}
				//System.out.println("in keyDown action is " + action);
			}
		});

	}
	
	private void initVars() {
		pos = 0.;
		posDot = 0.;
		angle = 0.;
		angleDot = 0.;
		action = 0;
	}

	public void start() {
		//Start animating!
		if (animatorThread == null) {
			animatorThread = new Thread(this);
		}
		animatorThread.start();
	}

	public void stop() {
		//Stop the animating thread.
		animatorThread = null;
		//Get rid of the objects necessary for double buffering.
		offGraphics = null;
		offImage = null;
	}

	public void updateState() {
		double velocity = posDot * tau;
		double[] rawStateVars = {angle, velocity};
		double[] stateVars = stateTranslator.toNN(rawStateVars);
		tmpCorrel.add(new double[] {action, angleDot * tau});
		
		if (!lastWasLoss) {
			modeler.observePostState(stateVars);
			modeler.saveMemory();
		}
		double[] nnAction = planner.getOptimalAction(stateVars, actionChoices,
				isSmart() ? 0.03 : 1, 0);
		action = (int) actionTranslator.fromNN(nnAction)[0];
		//			if (action == 0 && velocity == 0) action = Math.random() < 0.5 ? 1 : -1; // hack for random init
		modeler.observeAction(nnAction);
		modeler.observePreState(stateVars);
		
		lastWasLoss = false;
		
		// First calc derivatives of state variables
		double force = forceMag * action;
		double sinangle = Math.sin(angle); 
		double cosangle = Math.cos(angle);
		double angleDotSq = angleDot * angleDot;
		double common = (force + poleMassLength * angleDotSq * sinangle
				- fricCart * (posDot<0 ? -1 : 0)) / totalMass;
		double angleDDot = (9.8 * sinangle - cosangle * common
				- fricPole * angleDot / poleMassLength) /
				(halfPole * (fourthirds - poleMass * cosangle * cosangle /
						totalMass));
		double posDDot = common - poleMassLength * angleDDot * cosangle /
				totalMass;

		//Now update state.
		if (pos < -2.5) pos = 2.5;
		if (pos > 2.5) pos = -2.5;
		pos += posDot * tau;
		posDot += posDDot * tau;
		if (angle < -Math.PI) angle = Math.PI;
		if (angle > Math.PI) angle = -Math.PI;
		angle += angleDot * tau;
		angleDot += angleDDot * tau;

		frame++;
		if (Math.abs(angle) > 1.6) lose();
	}
	
	private boolean isSmart() {
		final double err = modeler.getTransitionsModule().getConfidenceEstimate();
				final int learnFrame = 1000;
		if (err > 0 && err < 0.001) return frame > learnFrame;
		else return false;
	}
	private void lose() {
		if (frame >= learnFrame && lastLoss < learnFrame) { // learn frame was between last loss and this
			modeler.learnFromMemory(1.5, 0.5, 0, false, 500, learnFrame, smartThresh);
			System.out.println("achieved confidence of " + modeler.getModelVTA().getConfidenceEstimate());
			learnFrame += learnFrameInc;
		}
		initVars();
		lastWasLoss = true;
		int thisLoss = frame;
		double vtaConf = modeler.getModelVTA().getConfidenceEstimate();
		System.out.println((thisLoss - lastLoss) + "	" + vtaConf + "	" + (isSmart()?"*":""));
		lastLoss = thisLoss;
		if (isSmart()) {
			for (double[] dd : tmpCorrel) System.out.println(dd[0] + "	" + dd[1]);
			modeler.testit(1000, stateMins, stateMaxes, stateTranslator, actionTranslator, actionChoices, false);
		}
	}

	public void run() {
		//Just to be nice, lower this thread's priority
		//so it can't interfere with other processing going on.
		//Thread.currentThread().setPriority(Thread.MIN_PRIORITY);

		//Remember the starting time.
		long startTime = System.currentTimeMillis();

		//This is the animation loop.
		while (Thread.currentThread() == animatorThread) {


			//Update the state of the pole; 
			updateState();

			//Display it.
			repaint();

			//Delay depending on how far we are behind.
			try {
				startTime += delay;
				Thread.sleep(Math.max(0, 
						startTime-System.currentTimeMillis()));
			} catch (InterruptedException e) {
				break;
			}
		}
	}

	public void paint(Graphics g) {
		update(g);
	}

	public void update(Graphics g) {
		Color bg = getBackground();
		Color fg = getForeground();
		Dimension d = getSize();
		Color cartColor = new Color(0,20,255);
		Color arrowColor = new Color(255,255,0);
		Color trackColor = new Color(100,100,50);

		//Create the offscreen graphics context, if no good one exists.
		if ( (offGraphics == null)
				|| (d.width != offDimension.width)
				|| (d.height != offDimension.height) ) {
			offDimension = d;
			offImage = createImage(d.width, d.height);
			offGraphics = offImage.getGraphics();
		}

		//Erase the previous image.
		offGraphics.setColor(getBackground());
		offGraphics.fillRect(0,0,d.width,d.height);

		//Draw Track.
		double xs[] = {-2.5, 2.5, 2.5, 2.3, 2.3, -2.3, -2.3, -2.5};
		double ys[] = {-0.4, -0.4, 0., 0., -0.2, -0.2, 0, 0};
		int pixxs[] = new int[8], pixys[] = new int[8];
		for (int i = 0; i<8; i++) {
			pixxs[i] = pixX(d,xs[i]);
			pixys[i] = pixY(d,ys[i]);
		}
		offGraphics.setColor(trackColor);
		offGraphics.fillPolygon(pixxs,pixys,8);

		//Draw message
		String msg = "Left Mouse Button: push left    Right Mouse Button: push right     Middle Button: PANIC";
		offGraphics.drawString(msg,20,d.height-20);

		//Draw cart.
		offGraphics.setColor(cartColor);
		offGraphics.fillRect(pixX(d,pos-0.2), pixY(d,0), pixDX(d,0.4), pixDY(d,-0.2));

		//Draw pole.
		//    offGraphics.setColor(cartColor);
		offGraphics.drawLine(pixX(d,pos),pixY(d,0),
				pixX(d,pos+Math.sin(angle)*poleLength),
				pixY(d,poleLength*Math.cos(angle)));

		//Draw action arrow.
		if (action != 0) {
			int signAction = (action > 0 ? 1 : (action < 0) ? -1 : 0);
			int tipx = pixX(d,pos+0.2*signAction);
			int tipy = pixY(d,-0.1);
			offGraphics.setColor(arrowColor);
			offGraphics.drawLine(pixX(d,pos),pixY(d,-0.1),tipx,tipy);
			offGraphics.drawLine(tipx,tipy,tipx-4*signAction,tipy+4);
			offGraphics.drawLine(tipx,tipy,tipx-4*signAction,tipy-4);
		}


		//Last thing: Paint the image onto the screen.
		g.drawImage(offImage, 0, 0, this);

	}

	public int pixX(Dimension d, double v) {
		return (int) Math.round((v + 2.5) / 5.0 * d.width);
	}

	public int pixY(Dimension d, double v) {
		return (int) Math.round(d.height - (v + 2.5) / 5.0 * d.height);
	}

	public int pixDX(Dimension d, double v) {
		return (int) Math.round(v / 5.0 * d.width);
	}

	public int pixDY(Dimension d, double v) {
		return (int) Math.round(-v / 5.0 * d.height);
	}

	public void resetPole() {
		pos = 0.;
		posDot = 0.;
		angle = 0.;
		angleDot = 0.;
	}
}
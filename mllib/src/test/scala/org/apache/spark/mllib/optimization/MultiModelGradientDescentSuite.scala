package org.apache.spark.mllib.optimization

import scala.collection.JavaConversions._
import scala.util.Random

import org.scalatest.{FunSuite, Matchers}

import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vectors}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.{LinearDataGenerator, LocalClusterSparkContext, LocalSparkContext}
import org.apache.spark.mllib.util.TestingUtils._

object MultiModelGradientDescentSuite {

  def generateLogisticInputAsList(
                                   offset: Double,
                                   scale: Double,
                                   nPoints: Int,
                                   seed: Int): java.util.List[LabeledPoint] = {
    seqAsJavaList(generateGDInput(offset, scale, nPoints, seed))
  }

  // Generate input of the form Y = logistic(offset + scale * X)
  def generateGDInput(
                       offset: Double,
                       scale: Double,
                       nPoints: Int,
                       seed: Int): Seq[LabeledPoint]  = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    val unifRand = new Random(45)
    val rLogis = (0 until nPoints).map { i =>
      val u = unifRand.nextDouble()
      math.log(u) - math.log(1.0-u)
    }

    val y: Seq[Int] = (0 until nPoints).map { i =>
      val yVal = offset + scale * x1(i) + rLogis(i)
      if (yVal > 0) 1 else 0
    }

    (0 until nPoints).map(i => LabeledPoint(y(i), Vectors.dense(x1(i))))
  }

  def generateSVMInputAsList(
                              intercept: Double,
                              weights: Array[Double],
                              nPoints: Int,
                              seed: Int): java.util.List[LabeledPoint] = {
    seqAsJavaList(generateSVMInput(intercept, weights, nPoints, seed))
  }

  // Generate noisy input of the form Y = signum(x.dot(weights) + intercept + noise)
  def generateSVMInput(
                        intercept: Double,
                        weights: Array[Double],
                        nPoints: Int,
                        seed: Int): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    val weightsMat = new DenseMatrix(weights.length, 1, weights)
    val x = Array.fill[Array[Double]](nPoints)(
      Array.fill[Double](weights.length)(rnd.nextDouble() * 2.0 - 1.0))
    val y = x.map { xi =>
      val yD = (new DenseMatrix(1, xi.length, xi) times weightsMat) +
        intercept + 0.01 * rnd.nextGaussian()
      if (yD.toArray(0) < 0) 0.0 else 1.0
    }
    y.zip(x).map(p => LabeledPoint(p._1, Vectors.dense(p._2)))
  }
}

class MultiModelGradientDescentSuite extends FunSuite with LocalSparkContext with Matchers {

  test("Assert the loss is decreasing.") {
    val nPoints = 10000
    val A = 2.0
    val B = -1.5

    val initialB = -1.0
    val initialWeights = Array(initialB)

    val gradient = new MultiModelLogisticGradient()
    val updater: Array[MultiModelUpdater] = Array(new MultiModelSimpleUpdater())
    val stepSize = Array(1.0, 0.1)
    val numIterations = Array(10)
    val regParam = Array(0.0)
    val miniBatchFrac = 1.0

    // Add a extra variable consisting of all 1.0's for the intercept.
    val testData = GradientDescentSuite.generateGDInput(A, B, nPoints, 42)
    val data = testData.map { case LabeledPoint(label, features) =>
      label -> Vectors.dense(1.0 +: features.toArray)
    }

    val dataRDD = sc.parallelize(data, 2).cache()
    val initialWeightsWithIntercept = Vectors.dense(1.0 +: initialWeights.toArray)

    val (_, loss) = MultiModelGradientDescent.runMiniBatchMMSGD(
      dataRDD,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFrac,
      initialWeightsWithIntercept)

    assert(loss.last(0) - loss.head(0) < 0, "loss isn't decreasing.")

    val lossDiff = loss.init.zip(loss.tail).map { case (lhs, rhs) => lhs(0) - rhs(0) }
    assert(lossDiff.count(_ > 0).toDouble / lossDiff.size > 0.8)
  }

  test("Test the loss and gradient of first iteration with regularization.") {

    val gradient = new MultiModelLogisticGradient()
    val updater: Array[MultiModelUpdater] = Array(new MultiModelSquaredL2Updater())

    // Add a extra variable consisting of all 1.0's for the intercept.
    val testData = GradientDescentSuite.generateGDInput(2.0, -1.5, 10000, 42)
    val data = testData.map { case LabeledPoint(label, features) =>
      label -> Vectors.dense(1.0 +: features.toArray)
    }

    val dataRDD = sc.parallelize(data, 2).cache()

    // Prepare non-zero weights
    val initialWeightsWithIntercept = Vectors.dense(1.0, 0.5)

    val regParam0 = Array(0.0)
    val (newWeights0, loss0) = MultiModelGradientDescent.runMiniBatchMMSGD(
      dataRDD, gradient, updater, Array(1.0), Array(1), regParam0, 1.0, initialWeightsWithIntercept)

    val regParam1 = Array(1.0)
    val (newWeights1, loss1) = MultiModelGradientDescent.runMiniBatchMMSGD(
      dataRDD, gradient, updater, Array(1.0), Array(1), regParam1, 1.0, initialWeightsWithIntercept)

    assert(
      loss1(0)(0) ~= (loss0(0)(0) + (math.pow(initialWeightsWithIntercept(0), 2) +
        math.pow(initialWeightsWithIntercept(1), 2)) / 2) absTol 1E-5,
      """For non-zero weights, the regVal should be \frac{1}{2}\sum_i w_i^2.""")

    assert(
      (newWeights1(0) ~= (newWeights0(0) - initialWeightsWithIntercept(0)) absTol 1E-5) &&
        (newWeights1(1) ~= (newWeights0(1) - initialWeightsWithIntercept(1)) absTol 1E-5),
      "The different between newWeights with/without regularization " +
        "should be initialWeightsWithIntercept.")
  }

  /*
  test("Check for correctness: LogisticRegression-SquaredL2Updater") {
    val nPoints = 10000
    val A = 2.0
    val B = -1.5

    val initialB = -1.0
    val initialWeights = Array(initialB)

    val gradient = new MultiModelLogisticGradient()
    val updater: Array[MultiModelUpdater] = Array(new MultiModelSquaredL2Updater())
    val stepSize = Array(1.0, 0.1)
    val numIterations = Array(100)
    val regParam = Array(0.0, 0.1, 1.0)
    val miniBatchFrac = 1.0

    // Add a extra variable consisting of all 1.0's for the intercept.
    val testData = GradientDescentSuite.generateGDInput(A, B, nPoints, 42)
    val data = testData.map { case LabeledPoint(label, features) =>
      label -> Vectors.dense(1.0 +: features.toArray)
    }

    val dataRDD = sc.parallelize(data, 2).cache()
    val initialWeightsWithIntercept = Vectors.dense(1.0 +: initialWeights.toArray.clone())

    val start1 = System.currentTimeMillis()
    val (weightsGD, _) = GradientDescent.runMiniBatchSGD(
      dataRDD,
      new LogisticGradient(),
      new SquaredL2Updater(),
      stepSize(0),
      numIterations(0),
      regParam(0),
      miniBatchFrac,
      initialWeightsWithIntercept)
    val dur1 = System.currentTimeMillis() - start1

    val start2 = System.currentTimeMillis()
    val forLoop = (0 until 6).map { i =>
      val (weightsGD2, _) = GradientDescent.runMiniBatchSGD(
        dataRDD,
        new LogisticGradient(),
        new SquaredL2Updater(),
        stepSize(math.round(i * 1.0 / 6).toInt),
        numIterations(0),
        regParam(i % 3),
        miniBatchFrac,
        Vectors.dense(1.0 +: initialWeights.toArray.clone()))
      weightsGD2
    }
    val dur2 = System.currentTimeMillis() - start2

    val res2 = Matrices.horzCat(forLoop.map( v => new DenseMatrix(v.size, 1, v.toArray)))

    val start3 = System.currentTimeMillis()
    val (weightsMMGD, _) = MultiModelGradientDescent.runMiniBatchMMSGD(
      dataRDD,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFrac,
      Vectors.dense(1.0 +: initialWeights.toArray))
    val dur3 = System.currentTimeMillis() - start3

    println(s"GD:\n$weightsGD\nTime: $dur1")
    println(s"MMGD:\n$weightsMMGD\nTime: $dur3")
    println(s"GD2:\n$res2\nTime: $dur2")

    assert(weightsGD(0) ~== weightsMMGD(0) absTol 1e-10)
    assert(weightsGD(1) ~== weightsMMGD(1) absTol 1e-10)

    assert(res2 ~== weightsMMGD absTol 1e-10)
  }


  test("Check for correctness: LinearRegression-SquaredL2Updater") {
    val nPoints = 10000
    val numFeatures = 50

    val initialWeights = Matrices.zeros(numFeatures, 1).toArray

    // Pick weights as random values distributed uniformly in [-0.5, 0.5]
    val w = Matrices.rand(numFeatures, 1) -= 0.5

    // Use half of data for training and other half for validation
    val data = LinearDataGenerator.generateLinearInput(0.0, w.toArray, nPoints, 42, 10.0)

    val gradient = new MultiModelLeastSquaresGradient()
    val updater: Array[MultiModelUpdater] = Array(new MultiModelSquaredL2Updater())
    val stepSize = Array(1.0, 0.1)
    val numIterations = Array(100)
    val regParam = Array(0.0, 0.1, 1.0)
    val miniBatchFrac = 1.0

    val dataRDD = sc.parallelize(data, 2).map( p => (p.label, p.features)).cache()

    val start = System.currentTimeMillis()
    val forLoop = (0 until 6).map { i =>
      val (weightsGD2, _) = GradientDescent.runMiniBatchSGD(
        dataRDD,
        new LeastSquaresGradient(),
        new SquaredL2Updater(),
        stepSize(math.round(i * 1.0 / 6).toInt),
        numIterations(0),
        regParam(i % 3),
        miniBatchFrac,
        Vectors.dense(initialWeights.clone()))
      weightsGD2
    }
    val dur = System.currentTimeMillis() - start

    val res = Matrices.horzCat(forLoop.map( v => new DenseMatrix(v.size, 1, v.toArray)))

    val start3 = System.currentTimeMillis()
    val (weightsMMGD, _) = MultiModelGradientDescent.runMiniBatchMMSGD(
      dataRDD,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFrac,
      Vectors.dense(initialWeights))
    val dur3 = System.currentTimeMillis() - start3

    println(s"Time: $dur3")
    println(s"Time: $dur")

    assert(res ~== weightsMMGD absTol 1e-10)
  }
  */
  test("Check for correctness: SVM-L1Updater") {
    val nPoints = 10000

    val initialWeights = Array(1.0, 0.0, 0.0)

    val A = 0.01
    val B = -1.5
    val C = 1.0

    val testData = MultiModelGradientDescentSuite.
      generateSVMInput(A, Array[Double](B, C), nPoints, 42)

    val data = testData.map { case LabeledPoint(label, features) =>
      label -> Vectors.dense(1.0 +: features.toArray)
    }

    val gradient = new MultiModelHingeGradient()
    val updater: Array[MultiModelUpdater] = Array(new MultiModelL1Updater())
    val stepSize = Array(1.0, 0.1)
    val numIterations = Array(10)
    val regParam = Array(0.0, 0.1)
    val miniBatchFrac = 1.0

    val dataRDD = sc.parallelize(data, 2).cache()

    dataRDD.count

    val numModels = stepSize.length * regParam.length * numIterations.length

    val start = System.currentTimeMillis()
    val forLoop = (0 until numModels).map { i =>
      val (weightsGD2, _) = GradientDescent.runMiniBatchSGD(
        dataRDD,
        new HingeGradient(),
        new L1Updater(),
        stepSize(math.round(i * 1.0 / numModels).toInt),
        numIterations(0),
        regParam(i % regParam.length),
        miniBatchFrac,
        Vectors.dense(initialWeights.clone()))
      weightsGD2
    }
    val dur = System.currentTimeMillis() - start

    val res = Matrices.horzCat(forLoop.map( v => new DenseMatrix(v.size, 1, v.toArray)))

    val start3 = System.currentTimeMillis()
    val (weightsMMGD, _) = MultiModelGradientDescent.runMiniBatchMMSGD(
      dataRDD,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFrac,
      Vectors.dense(initialWeights))
    val dur3 = System.currentTimeMillis() - start3

    println(s"Time: $dur3")
    println(s"Time: $dur")

    assert(res ~== weightsMMGD absTol 1e-10)
  }
}

/*
class MultiModelGradientDescentClusterSuite extends FunSuite with LocalClusterSparkContext {

  test("task size should be small") {
    val m = 4
    val n = 200000
    val points = sc.parallelize(0 until m, 2).mapPartitionsWithIndex { (idx, iter) =>
      val random = new Random(idx)
      iter.map(i => (1.0, Vectors.dense(Array.fill(n)(random.nextDouble()))))
    }.cache()
    // If we serialize data directly in the task closure, the size of the serialized task would be
    // greater than 1MB and hence Spark would throw an error.
    val (weights, loss) = MultiModelGradientDescent.runMiniBatchMMSGD(
      points,
      new MultiModelLogisticGradient,
      Array(new MultiModelSquaredL2Updater),
      Array(0.1),
      Array(2),
      Array(1.0),
      1.0,
      Vectors.dense(new Array[Double](n)))
  }
}
*/

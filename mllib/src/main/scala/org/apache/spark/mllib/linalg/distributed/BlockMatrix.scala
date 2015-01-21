/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.linalg.distributed

import breeze.linalg.{DenseMatrix => BDM}

import org.apache.spark.{Logging, Partitioner}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * A grid partitioner, which stores every block in a separate partition.
 *
 * @param numRowBlocks Number of blocks that form the rows of the matrix.
 * @param numColBlocks Number of blocks that form the columns of the matrix.
 */
private[mllib] class GridPartitioner(
    val numRowBlocks: Int,
    val numColBlocks: Int,
    val numParts: Int) extends Partitioner {
  // Having the number of partitions greater than the number of sub matrices does not help
  override val numPartitions = math.min(numParts, numRowBlocks * numColBlocks)

  /**
   * Returns the index of the partition the SubMatrix belongs to. Tries to achieve block wise
   * partitioning.
   *
   * @param key The key for the SubMatrix. Can be its position in the grid (its column major index)
   *            or a tuple of three integers that are the final row index after the multiplication,
   *            the index of the block to multiply with, and the final column index after the
   *            multiplication.
   * @return The index of the partition, which the SubMatrix belongs to.
   */
  override def getPartition(key: Any): Int = {
    key match {
      case (blockRowIndex: Int, blockColIndex: Int) =>
        getBlockId(blockRowIndex, blockColIndex)
      case (blockRowIndex: Int, innerIndex: Int, blockColIndex: Int) =>
        getBlockId(blockRowIndex, blockColIndex)
      case _ =>
        throw new IllegalArgumentException(s"Unrecognized key. key: $key")
    }
  }

  /** Partitions sub-matrices as blocks with neighboring sub-matrices. */
  private def getBlockId(blockRowIndex: Int, blockColIndex: Int): Int = {
    val totalBlocks = numRowBlocks * numColBlocks
    // Gives the number of blocks that need to be in each partition
    val partitionRatio = math.ceil(totalBlocks * 1.0 / numPartitions).toInt
    // Number of neighboring blocks to take in each row
    val subBlocksPerRow = math.ceil(numRowBlocks * 1.0 / partitionRatio).toInt
    // Number of neighboring blocks to take in each column
    val subBlocksPerCol = math.ceil(numColBlocks * 1.0 / partitionRatio).toInt
    // Coordinates of the block
    val i = blockRowIndex / subBlocksPerRow
    val j = blockColIndex / subBlocksPerCol
    val blocksPerRow = math.ceil(numRowBlocks * 1.0 / subBlocksPerRow).toInt
    j * blocksPerRow + i
  }

  /** Checks whether the partitioners have the same characteristics */
  override def equals(obj: Any): Boolean = {
    obj match {
      case r: GridPartitioner =>
        (this.numRowBlocks == r.numRowBlocks) && (this.numColBlocks == r.numColBlocks) &&
          (this.numPartitions == r.numPartitions)
      case _ =>
        false
    }
  }
}

/**
 * Represents a distributed matrix in blocks of local matrices.
 *
 * @param rdd The RDD of SubMatrices (local matrices) that form this matrix
 * @param nRows Number of rows of this matrix
 * @param nCols Number of columns of this matrix
 * @param numRowBlocks Number of blocks that form the rows of this matrix
 * @param numColBlocks Number of blocks that form the columns of this matrix
 * @param rowsPerBlock Number of rows that make up each block. The blocks forming the final
 *                     rows are not required to have the given number of rows
 * @param colsPerBlock Number of columns that make up each block. The blocks forming the final
 *                     columns are not required to have the given number of columns
 */
class BlockMatrix(
    val rdd: RDD[((Int, Int), Matrix)],
    private var nRows: Long,
    private var nCols: Long,
    val numRowBlocks: Int,
    val numColBlocks: Int,
    val rowsPerBlock: Int,
    val colsPerBlock: Int) extends DistributedMatrix with Logging {

  private type SubMatrix = ((Int, Int), Matrix) // ((blockRowIndex, blockColIndex), matrix)

  /**
   * Alternate constructor for BlockMatrix without the input of the number of rows and columns.
   *
   * @param rdd The RDD of SubMatrices (local matrices) that form this matrix
   * @param numRowBlocks Number of blocks that form the rows of this matrix
   * @param numColBlocks Number of blocks that form the columns of this matrix
   * @param rowsPerBlock Number of rows that make up each block. The blocks forming the final
   *                     rows are not required to have the given number of rows
   * @param colsPerBlock Number of columns that make up each block. The blocks forming the final
   *                     columns are not required to have the given number of columns
   */
  def this(
      rdd: RDD[((Int, Int), Matrix)],
      numRowBlocks: Int,
      numColBlocks: Int,
      rowsPerBlock: Int,
      colsPerBlock: Int) = {
    this(rdd, 0L, 0L, numRowBlocks, numColBlocks, rowsPerBlock, colsPerBlock)
  }

  private[mllib] var partitioner: GridPartitioner =
    new GridPartitioner(numRowBlocks, numColBlocks, rdd.partitions.length)

  private lazy val dims: (Long, Long) = getDim

  override def numRows(): Long = {
    if (nRows <= 0L) nRows = dims._1
    nRows
  }

  override def numCols(): Long = {
    if (nCols <= 0L) nCols = dims._2
    nCols
  }

  /** Returns the dimensions of the matrix. */
  private def getDim: (Long, Long) = {
    case class MatrixMetaData(var rowIndex: Int, var colIndex: Int,
        var numRows: Int, var numCols: Int)
    // picks the sizes of the matrix with the maximum indices
    def pickSizeByGreaterIndex(example: MatrixMetaData, base: MatrixMetaData): MatrixMetaData = {
      if (example.rowIndex > base.rowIndex) {
        base.rowIndex = example.rowIndex
        base.numRows = example.numRows
      }
      if (example.colIndex > base.colIndex) {
        base.colIndex = example.colIndex
        base.numCols = example.numCols
      }
      base
    }

    // Aggregate will return an error if the rdd is empty
    val lastRowCol = rdd.treeAggregate(new MatrixMetaData(0, 0, 0, 0))(
      seqOp = (c, v) => (c, v) match { case (base, ((blockXInd, blockYInd), mat)) =>
        pickSizeByGreaterIndex(
          new MatrixMetaData(blockXInd, blockYInd, mat.numRows, mat.numCols), base)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case (res1, res2) =>
          pickSizeByGreaterIndex(res1, res2)
      })
    // We add the size of the edge matrices, because they can be less than the specified
    // rowsPerBlock or colsPerBlock.
    (lastRowCol.rowIndex.toLong * rowsPerBlock + lastRowCol.numRows,
      lastRowCol.colIndex.toLong * colsPerBlock + lastRowCol.numCols)
  }

  /** Returns the Frobenius Norm of the matrix */
  def normFro(): Double = {
    math.sqrt(rdd.map { mat => mat._2 match {
      case sparse: SparseMatrix =>
        sparse.values.map(x => math.pow(x, 2)).sum
      case dense: DenseMatrix =>
        dense.values.map(x => math.pow(x, 2)).sum
      }
    }.reduce(_ + _))
  }

  /** Cache the underlying RDD. */
  def cache(): BlockMatrix = {
    rdd.cache()
    this
  }

  /** Set the storage level for the underlying RDD. */
  def persist(storageLevel: StorageLevel): BlockMatrix = {
    rdd.persist(storageLevel)
    this
  }

  /** Collect the distributed matrix on the driver as a `DenseMatrix`. */
  def toLocalMatrix(): Matrix = {
    require(numRows() < Int.MaxValue, "The number of rows of this matrix should be less than " +
      s"Int.MaxValue. Currently numRows: ${numRows()}")
    require(numCols() < Int.MaxValue, "The number of columns of this matrix should be less than " +
      s"Int.MaxValue. Currently numCols: ${numCols()}")
    val nRows = numRows().toInt
    val nCols = numCols().toInt
    val mem = nRows * nCols * 8 / 1000000
    if (mem > 500) logWarning(s"Storing this matrix will require $mem MB of memory!")

    val parts = rdd.collect().sortBy(x => (x._1._2, x._1._1))
    val values = new Array[Double](nRows * nCols)
    parts.foreach { case ((rowIndex, colIndex), block) =>
      val rowOffset = rowIndex * rowsPerBlock
      val colOffset = colIndex * colsPerBlock
      var j = 0
      val mat = block.toArray
      while (j < block.numCols) {
        var i = 0
        val indStart = (j + colOffset) * nRows + rowOffset
        val matStart = j * block.numRows
        while (i < block.numRows) {
          values(indStart + i) = mat(matStart + i)
          i += 1
        }
        j += 1
      }
    }
    new DenseMatrix(nRows, nCols, values)
  }

  /** Collects data and assembles a local dense breeze matrix (for test only). */
  private[mllib] def toBreeze(): BDM[Double] = {
    val localMat = toLocalMatrix()
    new BDM[Double](localMat.numRows, localMat.numCols, localMat.toArray)
  }
}

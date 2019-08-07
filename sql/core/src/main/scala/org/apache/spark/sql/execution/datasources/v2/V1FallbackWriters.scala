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

package org.apache.spark.sql.execution.datasources.v2

import scala.collection.JavaConverters._

import org.apache.spark.SparkException
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SaveMode}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.sources.{AlwaysTrue, CreatableRelationProvider, Filter, InsertableRelation}
import org.apache.spark.sql.sources.v2.Table
import org.apache.spark.sql.sources.v2.writer._
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/**
 * Physical plan node for append into a v2 table using V1 write interfaces.
 *
 * Rows in the output data set are appended.
 */
case class AppendDataExecV1(
    writeBuilder: V1WriteBuilder,
    plan: LogicalPlan) extends V1FallbackWriters {

  override protected def doExecute(): RDD[InternalRow] = {
    writeWithV1(writeBuilder.buildForV1Write())
  }
}

/**
 * Physical plan node for overwrite into a v2 table with V1 write interfaces. Note that when this
 * interface is used, the atomicity of the operation depends solely on the target data source.
 *
 * Overwrites data in a table matched by a set of filters. Rows matching all of the filters will be
 * deleted and rows in the output data set are appended.
 *
 * This plan is used to implement SaveMode.Overwrite. The behavior of SaveMode.Overwrite is to
 * truncate the table -- delete all rows -- and append the output data set. This uses the filter
 * AlwaysTrue to delete all rows.
 */
case class OverwriteByExpressionExecV1(
    table: Table,
    writeBuilder: V1WriteBuilder,
    deleteWhere: Array[Filter],
    plan: LogicalPlan) extends V1FallbackWriters {

  private def isTruncate(filters: Array[Filter]): Boolean = {
    filters.length == 1 && filters(0).isInstanceOf[AlwaysTrue]
  }

  override protected def doExecute(): RDD[InternalRow] = {
    writeBuilder match {
      case builder: SupportsTruncate if isTruncate(deleteWhere) =>
        writeWithV1(builder.truncate().asV1Writer.buildForV1Write())

      case builder: SupportsOverwrite =>
        writeWithV1(builder.overwrite(deleteWhere).asV1Writer.buildForV1Write())

      case _ =>
        throw new SparkException(s"Table does not support overwrite by expression: $table")
    }
  }
}

/** Some helper interfaces that use V2 write semantics through the V1 writer interface. */
sealed trait V1FallbackWriters extends SupportsV1Write {
  override def output: Seq[Attribute] = Nil
  override final def children: Seq[SparkPlan] = Nil

  protected implicit class toV1WriteBuilder(builder: WriteBuilder) {
    def asV1Writer: V1WriteBuilder = builder match {
      case v1: V1WriteBuilder => v1
      case other => throw new IllegalStateException(
        s"The returned writer ${other} was no longer a V1WriteBuilder.")
    }
  }
}

/**
 * A trait that allows Tables that use V1 Writer interfaces to append data.
 */
trait SupportsV1Write extends SparkPlan {
  def plan: LogicalPlan

  protected def writeWithV1(relation: InsertableRelation): RDD[InternalRow] = {
    relation.insert(Dataset.ofRows(sqlContext.sparkSession, plan), overwrite = false)
    sparkContext.emptyRDD
  }
}

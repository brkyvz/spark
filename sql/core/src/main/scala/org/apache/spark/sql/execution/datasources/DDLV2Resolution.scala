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

package org.apache.spark.sql.execution.datasources

import org.apache.spark.sql.catalog.v2._
import org.apache.spark.sql.catalyst.analysis.{NamedRelation, NoSuchDatabaseException, NoSuchTableException, UnresolvedRelation}
import org.apache.spark.sql.catalyst.catalog.{CatalogTableType, UnresolvedCatalogRelation}
import org.apache.spark.sql.catalyst.plans.logical.{AlterTable, LogicalPlan}
import org.apache.spark.sql.catalyst.plans.logical.sql._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.command.AlterTableAddColumnsCommand
import org.apache.spark.sql.execution.datasources.v2.{CatalogTableAsV2, DataSourceV2Relation}
import org.apache.spark.sql.internal.SQLConf

/**
 * Resolve ALTER TABLE statements that use a DSv2 catalog.
 *
 * This rule converts unresolved ALTER TABLE statements to v2 when a v2 catalog is responsible
 * for the table identifier. A v2 catalog is responsible for an identifier when the identifier
 * has a catalog specified, like prod_catalog.db.table, or when a default v2 catalog is set and
 * the table identifier does not include a catalog.
 */
class DDLV2Resolution(
    conf: SQLConf,
    lookup: LookupCatalog) extends Rule[LogicalPlan] {
  import org.apache.spark.sql.catalog.v2.CatalogV2Implicits._
  import lookup._

  private def getSessionCatalog: CatalogPlugin = lookup.sessionCatalog.getOrElse(
    throw new IllegalStateException("Session catalog not defined"))

  private def resolveTableRelation(
      catalog: TableCatalog,
      identifier: Identifier): Option[NamedRelation] = {
    val table = try catalog.loadTable(identifier) catch {
      case _: NoSuchDatabaseException => return None
      case _: NoSuchTableException => return None
    }
    table match {
      case CatalogTableAsV2(catalogTable) if catalogTable.tableType != CatalogTableType.VIEW =>
        Some(UnresolvedCatalogRelation(catalogTable))
      case CatalogTableAsV2(catalogTable) => None // must be a view
      case v2Table => Some(DataSourceV2Relation.create(v2Table))
    }
  }

  override def apply(plan: LogicalPlan): LogicalPlan = plan resolveOperators {
    case alter @ AlterTableAddColumnsStatement(
        CatalogObjectIdentifier(maybeCatalog, ident), cols) =>
      val catalog = maybeCatalog.getOrElse(getSessionCatalog).asTableCatalog

      resolveTableRelation(catalog, ident) match {
        case Some(namedRelation) =>
          val changes = cols.map { col =>
            TableChange.addColumn(col.name.toArray, col.dataType, true, col.comment.orNull)
          }
          AlterTable(catalog, ident, namedRelation, changes)
        case None =>
          alter
      }

    case alter @ AlterTableAlterColumnStatement(
        CatalogObjectIdentifier(maybeCatalog, ident), colName, dataType, comment) =>
      val catalog = maybeCatalog.getOrElse(getSessionCatalog).asTableCatalog

      resolveTableRelation(catalog, ident) match {
        case Some(namedRelation) =>
          val typeChange = dataType.map { newDataType =>
            TableChange.updateColumnType(colName.toArray, newDataType, true)
          }

          val commentChange = comment.map { newComment =>
            TableChange.updateColumnComment(colName.toArray, newComment)
          }
          AlterTable(catalog, ident, namedRelation, typeChange.toSeq ++ commentChange.toSeq)
        case None =>
          alter
      }

    case alter @ AlterTableRenameColumnStatement(
        CatalogObjectIdentifier(maybeCatalog, ident), col, newName) =>
      val catalog = maybeCatalog.getOrElse(getSessionCatalog).asTableCatalog

      resolveTableRelation(catalog, ident) match {
        case Some(namedRelation) =>
          AlterTable(
            catalog, ident, namedRelation, Seq(TableChange.renameColumn(col.toArray, newName)))
        case None =>
          alter
      }

    case alter @ AlterTableDropColumnsStatement(
        CatalogObjectIdentifier(maybeCatalog, ident), cols) =>
      val catalog = maybeCatalog.getOrElse(getSessionCatalog).asTableCatalog

      resolveTableRelation(catalog, ident) match {
        case Some(namedRelation) =>
          val changes = cols.map(col => TableChange.deleteColumn(col.toArray))
          AlterTable(catalog, ident, namedRelation, changes)
        case None =>
          alter
      }

    case alter @ AlterTableSetPropertiesStatement(
        CatalogObjectIdentifier(maybeCatalog, ident), props) =>
      val catalog = maybeCatalog.getOrElse(getSessionCatalog).asTableCatalog

      resolveTableRelation(catalog, ident) match {
        case Some(namedRelation) =>
          val changes = props.map {
            case (key, value) =>
              TableChange.setProperty(key, value)
          }
          AlterTable(catalog, ident, namedRelation, changes.toSeq)
        case None =>
          alter
      }

    case alter @ AlterTableUnsetPropertiesStatement(
        CatalogObjectIdentifier(maybeCatalog, ident), keys, _) =>
      val catalog = maybeCatalog.getOrElse(getSessionCatalog).asTableCatalog

      resolveTableRelation(catalog, ident) match {
        case Some(namedRelation) =>
          val changes = keys.map(key => TableChange.removeProperty(key))
          AlterTable(catalog, ident, namedRelation, changes)
        case None =>
          alter
      }

    case alter @ AlterTableSetLocationStatement(
        CatalogObjectIdentifier(maybeCatalog, ident), newLoc) =>
      val catalog = maybeCatalog.getOrElse(getSessionCatalog).asTableCatalog

      resolveTableRelation(catalog, ident) match {
        case Some(namedRelation) =>
          val changes = Seq(TableChange.setProperty("location", newLoc))
          AlterTable(catalog, ident, namedRelation, changes)
        case None =>
          alter
      }
  }
}

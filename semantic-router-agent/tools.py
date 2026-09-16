"""Invoice and catalog tools adapted from the langchain-basics music store."""

from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from pydantic import Field

from db import query


@dataclass
class UserContext:
    customer_id: int | None = None
    # The sample invoices end in December 2025; this is the demo's reference date.
    as_of: str = "2025-12-22"


Months = Annotated[int, Field(ge=1, le=12, strict=True)]
Limit = Annotated[int, Field(ge=1, le=50, strict=True)]
InvoiceId = Annotated[int, Field(gt=0, strict=True)]


def _customer_context(runtime: ToolRuntime[UserContext]) -> UserContext | str:
    context = runtime.context
    if context is None or context.customer_id is None:
        return "A customer must be supplied in the demo's runtime context."
    if type(context.customer_id) is not int or context.customer_id <= 0:
        return "The runtime customer ID must be a positive integer."
    try:
        if date.fromisoformat(context.as_of).isoformat() != context.as_of:
            raise ValueError
    except (TypeError, ValueError):
        return "The runtime as_of date must use YYYY-MM-DD format."
    return context


@tool
async def list_my_invoices(
    runtime: ToolRuntime[UserContext],
    months: Months | None = None,
) -> dict:
    """List the current customer's invoices through the demo reference date.

    Omit months for all recorded invoices. Set months=3 to retrieve the last
    three calendar months, ending on the reference date (both dates included).
    """
    context = _customer_context(runtime)
    if isinstance(context, str):
        return {"error": context}

    start_date = None
    if months is not None:
        as_of = date.fromisoformat(context.as_of)
        year, month_index = divmod(as_of.year * 12 + as_of.month - 1 - months, 12)
        if year < 1:
            return {"error": "The requested window starts before year 0001."}
        month = month_index + 1
        day = min(as_of.day, monthrange(year, month)[1])
        start_date = date(year, month, day).isoformat()

    invoices = await query(
        """
        SELECT InvoiceId AS invoice_id, date(InvoiceDate) AS date, Total AS total
        FROM Invoice
        WHERE CustomerId = :customer_id
          AND date(InvoiceDate) <= :as_of
          AND (:start_date IS NULL OR date(InvoiceDate) >= :start_date)
        ORDER BY InvoiceDate DESC, InvoiceId DESC
        """,
        {
            "customer_id": context.customer_id,
            "as_of": context.as_of,
            "start_date": start_date,
        },
    )
    return {
        "as_of": context.as_of,
        "window_start": start_date,
        "window_end": context.as_of,
        "invoices": invoices,
    }


@tool
async def get_invoice_details(
    invoice_id: InvoiceId,
    runtime: ToolRuntime[UserContext],
) -> dict:
    """Get purchased tracks, artists, albums, genres, and prices on your invoice."""
    context = _customer_context(runtime)
    if isinstance(context, str):
        return {"error": context}

    parameters = {
        "invoice_id": invoice_id,
        "customer_id": context.customer_id,
        "as_of": context.as_of,
    }
    invoices = await query(
        """
        SELECT InvoiceId AS invoice_id, date(InvoiceDate) AS date, Total AS total
        FROM Invoice
        WHERE InvoiceId = :invoice_id AND CustomerId = :customer_id
          AND date(InvoiceDate) <= :as_of
        """,
        parameters,
    )
    if not invoices:
        return {"error": "This invoice is not available for your account and demo date."}

    lines = await query(
        """
        SELECT t.Name AS track, ar.Name AS artist, a.Title AS album,
               g.Name AS genre, mt.Name AS media_type,
               il.UnitPrice AS unit_price, il.Quantity AS quantity
        FROM InvoiceLine il
        JOIN Invoice i ON i.InvoiceId = il.InvoiceId
        JOIN Track t ON t.TrackId = il.TrackId
        JOIN Album a ON a.AlbumId = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = a.ArtistId
        JOIN Genre g ON g.GenreId = t.GenreId
        JOIN MediaType mt ON mt.MediaTypeId = t.MediaTypeId
        WHERE i.InvoiceId = :invoice_id AND i.CustomerId = :customer_id
          AND date(i.InvoiceDate) <= :as_of
        ORDER BY il.InvoiceLineId
        """,
        parameters,
    )
    return {"as_of": context.as_of, **invoices[0], "items": lines}


@tool
async def popular_in_genre(
    genre: str,
    runtime: ToolRuntime[UserContext],
    limit: Limit = 5,
) -> dict:
    """Find popular catalog music in a genre, excluding tracks you already bought.

    Use genre names returned by invoice details. Results exclude videos and
    are ranked by sales through the demo reference date.
    """
    context = _customer_context(runtime)
    if isinstance(context, str):
        return {"error": context}

    tracks = await query(
        """
        SELECT t.Name AS track, ar.Name AS artist, a.Title AS album,
               g.Name AS genre, t.UnitPrice AS unit_price,
               COUNT(sale.InvoiceId) AS times_sold
        FROM Track t
        JOIN Genre g ON g.GenreId = t.GenreId
        JOIN Album a ON a.AlbumId = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = a.ArtistId
        JOIN MediaType mt ON mt.MediaTypeId = t.MediaTypeId
        LEFT JOIN InvoiceLine il ON il.TrackId = t.TrackId
        LEFT JOIN Invoice sale ON sale.InvoiceId = il.InvoiceId
                              AND date(sale.InvoiceDate) <= :as_of
        WHERE g.Name = :genre COLLATE NOCASE
          AND mt.Name NOT LIKE '%video%'
          AND NOT EXISTS (
              SELECT 1
              FROM InvoiceLine purchased
              JOIN Invoice owned ON owned.InvoiceId = purchased.InvoiceId
              WHERE purchased.TrackId = t.TrackId
                AND owned.CustomerId = :customer_id
                AND date(owned.InvoiceDate) <= :as_of
          )
        GROUP BY t.TrackId
        ORDER BY times_sold DESC, t.Name, t.TrackId
        LIMIT :limit
        """,
        {
            "genre": genre.strip(),
            "customer_id": context.customer_id,
            "as_of": context.as_of,
            "limit": limit,
        },
    )
    return {"as_of": context.as_of, "genre": genre.strip(), "tracks": tracks}

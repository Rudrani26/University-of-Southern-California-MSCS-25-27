from setup import con, print_table

query = """
SELECT  
    p.PersonID,
    p.FName || ' ' || p.LName AS Instructor,
    ROUND(AVG(CAST(r.Stars AS DOUBLE)), 2) AS AvgRating,
    COUNT(*) AS RatingCount
FROM    PERSON p
JOIN    RATING_INSTRUCTOR r ON r.PersonID = p.PersonID
GROUP BY p.PersonID, p.FName, p.LName
ORDER BY AvgRating DESC, RatingCount DESC
LIMIT 1;
"""

result = con.execute(query).fetchdf()
print("Q3. who is the most popular instructor (ie. who has the highest rating)?")
print_table(result)

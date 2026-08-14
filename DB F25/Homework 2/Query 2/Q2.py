from setup import con, print_table

query = """
SELECT P.PersonID,
       P.FName || ' ' || P.LName AS Instructor,
       COUNT(DISTINCT E.StudentID) AS StudentsTaught
FROM PERSON P
JOIN CODINGCLASS CL ON CL.FacilitatorID = P.PersonID
JOIN GROUP_ENROLMENT E ON E.GroupID = CL.ClassID
GROUP BY P.PersonID, P.FName, P.LName
ORDER BY StudentsTaught DESC
LIMIT 1;
"""

result = con.execute(query).fetchdf()
print("Q2. Who is the most popular instructor (ie. who teaches the most number of students)?")
print_table(result)

from setup import con, print_table

query = """
SELECT C.CourseID,
       C.Title,
       COUNT(DISTINCT E.StudentID) AS StudentCount
FROM COURSE C
JOIN CODINGCLASS CL ON CL.CourseID = C.CourseID
JOIN GROUP_ENROLMENT E ON E.GroupID = CL.ClassID
GROUP BY C.CourseID, C.Title
ORDER BY StudentCount DESC
LIMIT 1;
"""

result = con.execute(query).fetchdf()
print("Q1. What is the course with most number of students?")
print_table(result)

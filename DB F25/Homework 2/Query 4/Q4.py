from setup import con, print_table

# Query 4
query ="""
SELECT  c.Title AS class_name,
        COUNT(DISTINCT e.StudentID) AS students_enrolled
FROM    COURSE c
JOIN    CODINGCLASS cl ON cl.CourseID = c.CourseID
JOIN    GROUP_ENROLMENT e ON e.GroupID = cl.ClassID
GROUP BY c.CourseID, c.Title
ORDER BY students_enrolled DESC;
"""

result = con.execute(query).fetchdf()
print("Q4. create a listing that includes class name and the number of students enrolled in the class, sorted in reverse order of enrollment (eg. to tell which were the most popular classes, at the end of the term)")
print_table(result)

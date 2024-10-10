from streamlit_agraph import agraph, Node, Edge, Config
import streamlit as st


st.write('US Market only')
nodes = []
edges = []



nodes.append( Node(id="Market", 
                   size=25,
                   shape="start",
                   label="Market",
                   image=""
                   ) 
            )


nodes.append( Node(id="Utilities", 
                   label="Utilities", 
                   size=25, 
                   shape="start",
                   image="") 
            ) # includes **kwargs
nodes.append( Node(id="Real Estate", 
                   size=25,
                   shape="circularImage",
                   label="Real Estate",
                   image="",) 
            )

nodes.append( Node(id="Consumer Staples", 
                   size=25,
                   shape="circularImage",
                   label="Consumer Staples",
                   image=""
                   ) 
            )

nodes.append( Node(id="Consumer Discrentionary", 
                   size=25,
                   shape="circularImage",
                   label="Consumer Discrentionary",
                   image=""
                   ) 
            )
nodes.append( Node(id="Energy", 
                   size=25,
                   shape="circularImage",
                   label="Energy",
                   image=""
                   ) 
            )
nodes.append( Node(id="Communication Services", 
                   size=25,
                   shape="circularImage",
                   label="Communication Services",
                   image=""
                   ) 
            )

nodes.append( Node(id="Industrials", 
                   size=25,
                   shape="circularImage",
                   label="Industrials",
                   image=""
                   ) 
            )

nodes.append( Node(id="Finance", 
                   size=25,
                   shape="circularImage",
                   label="Finance",
                   image=""
                   ) 
            )


nodes.append( Node(id="Basic Materials", 
                   size=25,
                   shape="circularImage",
                   label="Basic Materials",
                   image=""
                   ) 
            )

nodes.append( Node(id="Technology", 
                   size=25,
                   shape="circularImage",
                   label="Technology",
                   image=""
                   ) 
            )

edges.append( Edge(source="Real Estate", 
                   label="", 
                   target="Utilities", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 

edges.append( Edge(source="Consumer Staples", 
                   label="", 
                   target="Utilities", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 

edges.append( Edge(source="Consumer Discrentionary", 
                   label="", 
                   target="Market", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 


edges.append( Edge(source="Communication Services", 
                   label="", 
                   target="Market", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 

edges.append( Edge(source="Market", 
                   label="", 
                   target="Communication Services", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )


edges.append( Edge(source="Market", 
                   label="", 
                   target="Energy", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 



edges.append( Edge(source="Industrials", 
                   label="", 
                   target="Finance", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 



edges.append( Edge(source="Basic Materials", 
                   label="", 
                   target="Industrials", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )
edges.append( Edge(source="Finance", 
                   label="", 
                   target="Industrials", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )  
edges.append( Edge(source="Consumer Discretionary", 
                   label="", 
                   target="Industrials", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )  

edges.append( Edge(source="Industrials", 
                   label="", 
                   target="Basic Materials", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )  

edges.append( Edge(source="Industrials", 
                   label="", 
                   target="Technology",
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )  
edges.append( Edge(source="Market", 
                   label="", 
                   target="Technology", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 

edges.append( Edge(source="Utilities", 
                   label="", 
                   target="Real Estate", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )
edges.append( Edge(source="Communication Services", 
                   label="", 
                   target="Real Estate",
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )
edges.append( Edge(source="Market", 
                   label="", 
                   target="Consumer Discretionary",
                   color='green' ,smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )

#########
#Staples
############

edges.append( Edge(source="Basic Materials", 
                   label="", 
                   target="Consumer Staples", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )
edges.append( Edge(source="Consumer Discrentionary", 
                   label="", 
                   target="Consumer Staples", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 
edges.append( Edge(source="Utilities", 
                   label="", 
                   target="Consumer Staples", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )



config = Config(width=1000,
                height=1000,
                directed=True, 
                physics=True, 
                hierarchical=False,

                # **kwargs
                )

return_value = agraph(nodes=nodes, 
                      edges=edges, 
                      config=config)

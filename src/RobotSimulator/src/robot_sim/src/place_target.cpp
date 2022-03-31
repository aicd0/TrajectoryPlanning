#include <ros/ros.h>
#include <gazebo_msgs/GetLinkProperties.h>
#include <gazebo_msgs/SetLinkProperties.h>
#include <geometry_msgs/Point.h>
#include <string>
#include "robot_sim/PlaceTarget.h"

using namespace std;

const string target_link_name = "target";
ros::ServiceClient get_link_property_client;
ros::ServiceClient set_link_property_client;

bool callback(robot_sim::PlaceTarget::Request  &req,
              robot_sim::PlaceTarget::Response &res)
{
  gazebo_msgs::GetLinkProperties get_link_req;
  get_link_req.request.link_name = target_link_name;
  get_link_property_client.call(get_link_req);

  gazebo_msgs::SetLinkProperties set_link_req;
  set_link_req.request.link_name = target_link_name;
  set_link_req.request.com.orientation = get_link_req.response.com.orientation;
  set_link_req.request.com.position.x = -req.position.x;
  set_link_req.request.com.position.y = -req.position.y;
  set_link_req.request.com.position.z = -req.position.z;
  set_link_req.request.gravity_mode = get_link_req.response.gravity_mode;
  set_link_req.request.ixx = get_link_req.response.ixx;
  set_link_req.request.ixy = get_link_req.response.ixy;
  set_link_req.request.ixz = get_link_req.response.ixz;
  set_link_req.request.iyy = get_link_req.response.iyy;
  set_link_req.request.iyz = get_link_req.response.iyz;
  set_link_req.request.izz = get_link_req.response.izz;
  set_link_req.request.mass = get_link_req.response.mass;
  set_link_property_client.call(set_link_req);
  return true;
}

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "place_target_node");
  ros::NodeHandle nh;

  {
    const string srv_name = "/gazebo/get_link_properties";
    ros::service::waitForService(srv_name);
    get_link_property_client = nh.serviceClient<gazebo_msgs::GetLinkProperties>(srv_name);
  }

  {
    const string srv_name = "/gazebo/set_link_properties";
    ros::service::waitForService(srv_name);
    set_link_property_client = nh.serviceClient<gazebo_msgs::SetLinkProperties>(srv_name);
  }

  auto service = nh.advertiseService("/user/place_target", callback);
  ros::spin();
}
